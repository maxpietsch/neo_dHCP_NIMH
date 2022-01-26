import os, json, sys, re
import pandas as pd
from pathlib import Path
import argparse

from tqdm.auto import tqdm
import pprint

import hashlib
from datetime import datetime
from nda_manifests import *

import nibabel as nib

DEFAULT_PARAMS = {
    'scanner_manufacturer_pd': "Philips Medical Systems",
    'scanner_type_pd': "Achieva",
    'scanner_software_versions_pd': "3.2.2\3.2.2.0",
    'magnetic_field_strength': 3,
    'mri_repetition_time_pd': None,
    'mri_echo_time_pd': None,
    'flip_angle': None,
    'acquisition_matrix': None,
    'mri_field_of_view_pd': None,
    'patient_position': "HFS",
    'photomet_interpret': "MONOCHROME2",
    'image_num_dimensions': None,
    'image_extent1': None,
    'image_unit1': None,
    'image_resolution1': None,
    'image_slice_thickness': None,
    'image_orientation': 'axial',
    'slice_timing': None,
    'image_extent2': None,
    'image_extent3': None,
    # 'image_extent4': None,
    'image_resolution2': None,
    'image_resolution3': None,
    # 'image_resolution4': None,
    # 'image_resolution5': None
}

d_PARAMKEY_JSONKEY = {
    'scanner_manufacturer_pd': "Manufacturer",
    'scanner_type_pd': "ManufacturersModelName",
    'scanner_software_versions_pd': "SoftwareVersions",
    'magnetic_field_strength': "MagneticFieldStrength",
    'mri_repetition_time_pd': "RepetitionTime",
    'mri_echo_time_pd': "EchoTime",
    'flip_angle': "FlipAngle",
    'slice_timing': "SliceTiming",
}


d_MODALITY_MAINFILEPATTERN = {
    'func': '_task-rest_bold',
    'B1': '_magnitude',
    'anat-T2': 'rec-SVR_T2w',
    'anat-T1': 'rec-SVR_T1w',
    'dwi': re.compile('.*_run-\d\d_dwi$'),
    'fmap': '_magnitude'
}


def filter_stem(modal, d_imagestem_suffixes):
    """ get main image file for modality `modal` to parse json and header details from """

    assert modal in d_MODALITY_MAINFILEPATTERN, modal
    for stem in sorted(d_imagestem_suffixes):
        pattern = d_MODALITY_MAINFILEPATTERN[modal]
        if isinstance(pattern, str) and stem.endswith(pattern):
            return stem
        elif isinstance(pattern, re.Pattern) and pattern.match(stem):
            return stem


class ImageMetadataParser:
    def __init__(self):
        self.image_suffixes = ['.nii.gz', '.nii', '.json']

    def get_image_data(self, manifest):
        d_image_suf = {}
        for p in manifest['files']:
            p = p['path']
            for suf in self.image_suffixes:
                if p.endswith(suf):
                    stem = p[:-len(suf)]
                    if stem not in d_image_suf:
                        d_image_suf[stem] = []
                    d_image_suf[stem] += [suf]
                    break
        return d_image_suf


    def get_image_parameters(self, modal, manifest):
        params = {k:v for k, v in DEFAULT_PARAMS.items()}

        d_image_suf = self.get_image_data(manifest)
        stem = filter_stem(modal, d_image_suf)
        print(stem)
        assert stem is not None, f'modality {modal} has no image files in {sorted(d_image_suf.keys())}'
        suffs = d_image_suf[stem]
        # parse json
        js = {}
        if '.json' in suffs:
            with open(stem+'.json', 'r') as f:
                js = json.load(f)
            for pk, jk in d_PARAMKEY_JSONKEY.items():
                if jk in js:
                    params[pk] = js[jk]
            if js.get('AcqVoxelSize', None) is not None:
                params['image_slice_thickness'] = js['AcqVoxelSize'][2] #assumes image_orientation == axial

        missing = {k:v for k, v in params.items() if v is None}
        # parse image header
        if missing:
            ext = None
            for suf in suffs:
                if '.nii' in suf:
                    ext = suf
                    break
            if ext is None:
                raise IOError(str(stem)+' is missing .nii image: '+str(suffs))
            img = nib.load(stem + ext)
            voxel_grid = img.header.get_data_shape()
            voxel_res = img.header.get_zooms()
            units = img.header.get_xyzt_units()
            fov = []
            for dim, (size, res) in enumerate(zip(voxel_grid, voxel_res), 1):
                params[f'image_extent{dim}'] = size
                params[f'image_resolution{dim}'] = res
                fov.append(size*res)
                if dim <= 3:
                    if units[0] == 'unknown':
                        params[f'image_unit{dim}'] = 'Millimeters'
                    else:
                        params[f'image_unit{dim}'] = units[0]
            params['acquisition_matrix'] = list(voxel_grid[:2])
            if params.get('image_slice_thickness', None) is not None and params['image_slice_thickness'] != voxel_res[2]:
                print(f"using AcqVoxelSize {params['image_slice_thickness']}, not image voxel size {voxel_res[2]} for slice thickness")
            else:
                print(f"using image voxel size {voxel_res[2]} for slice thickness")
                params['image_slice_thickness'] = voxel_res[2]
            params['image_num_dimensions'] = dim
            params['mri_field_of_view_pd'] = fov[:3]

        missing = {k:v for k, v in params.items() if v is None}
        if missing:
            print('missing parameters:\n'+pprint.pformat(missing))

        return params


class Image03Gen:
    def __init__(self,
                 d_sub_guid,
                 d_subses_gascan,
                 d_subses_sex,
                 image03_template,
                 image03_definitions
                 ):
        self.d_sub_guid = d_sub_guid
        self.d_subses_gascan = d_subses_gascan
        self.d_subses_sex = d_subses_sex

        self.df_template = pd.read_csv(image03_template)
        self.df_definitions = pd.read_csv(image03_definitions)

        self.required = self.df_definitions.loc[self.df_definitions.Required == 'Required', :]
        self.conditional = self.df_definitions.loc[self.df_definitions.Required == 'Conditional', :]

        self.Hash = hashlib.sha512
        self.MAX_HASH_PLUS_ONE = 2**(self.Hash().digest_size * 8)

    @staticmethod
    def ga_to_months(ga_weeks):
        """ convert gestational age in weeks to age in months.
        NDA rule: Age is rounded to chronological month. If the research participant is 15-days-old at time of interview,
        the appropriate value would be 0 months. If the participant is 16-days-old, the value would be 1 month."""
        return int(round(ga_weeks/4.0, 0))

    def str_to_probability(self, in_str):
        """Return a reproducible uniformly random float in the interval [0, 1) for the given string."""
        seed = in_str.encode()
        hash_digest = self.Hash(seed).digest()
        hash_int = int.from_bytes(hash_digest, 'big')  # Uses explicit byteorder for system-agnostic reproducibility
        return hash_int / self.MAX_HASH_PLUS_ONE  # Float division

    def str_to_random_date(self, in_str):
        val = self.str_to_probability(in_str)
        start = datetime(1970,1,1,0,0,0)
        end = datetime(2000,1,1,0,0,0)
        val = datetime.fromtimestamp(start.timestamp() + val * (end.timestamp() - start.timestamp()))
        return val.strftime('%m/%d/%Y')

    @staticmethod
    def warn(*args, **kwargs):
        print(*args, **kwargs)

    def make_row(self, sub, ses, **kwargs):
        subses = f"{sub}-{ses}"
        defaults = {'transformation_performed':'No', 'image_file_format':'NIFTI', 'scan_object':'Live', 'image_modality':'MRI'}
        defaults.update(kwargs)
        image03_row = {}
        for _, row in self.required.iterrows():
            element = row['ElementName']
            if element == 'subjectkey':
                val = defaults.get('subjectkey', self.d_sub_guid[sub])
            elif element == 'src_subject_id':
                val = defaults.get('src_subject_id', sub)
            elif element == 'sex':
                val = defaults.get('sex', self.d_subses_sex.get(subses, None))
            elif element == 'interview_age':
                val = defaults.get('interview_age', None)
                if val is None and subses in self.d_subses_gascan:
                    val = self.ga_to_months(self.d_subses_gascan.get(subses))
            elif element == 'interview_date':
                val = defaults.get(element, None)
                if val is None: # make up random but reproducible date
                    val = self.str_to_random_date(subses)
            else:
                val = defaults.get(element, None)

            if val is None:
                self.warn(f'{subses}: missing value for {element}')
            image03_row[element] = val

            # check valid values
            if isinstance(row['ValueRange'], str) and ';' in row['ValueRange']:
                valid = [v.lstrip().rstrip() for v in row['ValueRange'].split(';')]
                if image03_row[element] not in valid:
                    raise ValueError(f'{element}:{image03_row[element]} is not valid, needs to be one of {valid}')
            if isinstance(row['ValueRange'], str) and '::' in row['ValueRange']:
                lo, hi = map(int, row['ValueRange'].split('::'))
                if not isinstance(val, int) or lo > val or hi < val:
                    raise ValueError(f'{element}:{image03_row[element]} is not valid, needs to be integer in range {lo} {hi}\n{image03_row}')

        # add non-required
        for k, v in kwargs.items():
            if k not in image03_row:
                image03_row[k] = v

        # check most simple conditionals (incomplete)
        for _, row in self.conditional.iterrows():
            element = row['ElementName']
            condition = row['Condition']
            if condition.count('==') == 1:
                k, v = condition.split('==')
                k = k.lstrip().rstrip()
                v = v.replace("'", "").lstrip().rstrip()
                if k in image03_row and image03_row[k] == v:
                    if element not in image03_row:
                        raise ValueError(f'required field {element} for {k}=={v}:\n{row}\n{image03_row}')
        # TODO check all conditionals (evaluate logic via pandas query?)
        if 'diffusion' in image03_row.get('scan_type', "") and image03_row.get('bvek_bval_files', None) != 'Yes':
            raise ValueError(f"scan_type=={image03_row['scan_type']}: set bvek_bval_files to 'Yes'\n{image03_row}")

        return image03_row


class RawParser:
    """
    captures file tree by subject, session and modality as manifest for use in image03
    data on disk:
    rel3_derivatives/rel3_rawdata_vol?/sub-*/ses-*/<modality>/file tree
    caches manifest files in `manifest_dir`
    """
    def __init__(self, toplevel, manifest_dir, only_subject=None):
        self.manifests = {}
        pbar = tqdm([d for d in self.listdirs(toplevel) if d.startswith('sub-')])
        for sub in pbar:

            if only_subject is not None and sub != only_subject:
                continue

            session_ids = pd.read_csv(toplevel / sub / f'{sub}_sessions.tsv' , delimiter='\t')['session_id'].values.tolist()
            sessions = [f'ses-{ses}' for ses in session_ids]
            # TODO make sure subfolders match sessions

            for ses in sessions:
                for modality in self.listdirs(toplevel / sub / ses):
                    input_dir = toplevel / sub / ses / modality
                    manifest_json = manifest_dir / f'{sub}-{ses}_raw{modality}.json'
                    if not os.path.isfile(manifest_json):
                        pbar.set_description(f"Processing {manifest_json}")
                        manifest = self.make_manifest(input_dir)
                        manifest.output_as_file(manifest_dir / f'{sub}-{ses}_raw{modality}.json')
                    with open(manifest_json, 'r') as f:
                        manifest = json.load(f)
                        self.manifests[(sub, ses, modality, input_dir, manifest_json)] = manifest

    @staticmethod
    def make_manifest(input_dir):
        manifest = Manifest()
        manifest.create_from_dir(input_dir)
        return manifest

    @staticmethod
    def listdirs(toplevel):
        toplevel = Path(toplevel)
        return sorted(filter(lambda x: os.path.isdir(toplevel / x), os.listdir(toplevel)))



def load_args():
    parser = argparse.ArgumentParser(description='Create a Manifest Files and Image03 from a directory ')
    parser.add_argument('-id', '--input-dir', help='A directory as input, to create Manifest Files from.')
    args = parser.parse_args()

    if args.input_dir is None:
        parser.error('No input dir specified')
    return args


if __name__ == '__main__':
    pd.options.display.max_rows = 999
    pd.set_option("max_colwidth", 80)

    args = load_args()

    image03_template = 'templates/image03_template.csv'
    image03_definitions = 'templates/image03_definitions.csv'

    url = 'https://raw.githubusercontent.com/BioMedIA/dHCP-release-notes/master/supplementary_files/combined.tsv'
    df_neo = pd.read_csv(url, sep='\t')
    d_subses_gascan = {}
    d_subses_sex = {}
    for _, row in df_neo.iterrows():
        sub = 'sub-'+row['participant_id']
        ses = 'ses-'+str(int(row['session_id']))
        d_subses_gascan[sub+'-'+ses] = row['scan_age']
        d_subses_sex[sub+'-'+ses] = {'male':'M', 'female':'F'}.get(row['sex'])

    d_sub_guid = {}
    with open('dhcp/GUID_mapping_all', 'r') as f:
        for l in f.readlines():
            l = l.split()
            if l:
                try:
                    sub, _, guid = l
                except:
                    raise IOError('GUID mapping not parsable')
                if not sub.startswith('sub-'):
                    sub = 'sub-' + sub
                d_sub_guid[sub] = guid


    manifest_dir = Path('manifests')

    d_modality_scantype = {'anat': 'MR structural (T1, T2)',
                           'B1': 'MR structural (B1 map)',
                           'dwi': 'MR diffusion',
                           'fmap': 'Field Map',
                           'func': 'fMRI'
                           }
    rows = []

    image03 = Image03Gen(d_sub_guid,
                         d_subses_gascan=d_subses_gascan,
                         d_subses_sex=d_subses_sex,
                         image03_template=image03_template,
                         image03_definitions=image03_definitions)


    toplevel = Path(args.input_dir)

    raw = RawParser(toplevel, manifest_dir)

    for (sub, ses, modality, input_dir, manifest_json), manifest in raw.manifests.items():
        print(input_dir)
        # check if all files are accounted for
        all_files = []
        for path, subdirs, files in os.walk(input_dir):
            for name in files:
                f = os.path.join(path, name)
                if os.path.isfile(f):
                    all_files.append(Path(f).absolute())
        manifest_files = set([Path(f['path']).absolute() for f in manifest['files']])
        all_files = set(all_files)
        assert len(all_files - manifest_files) == 0
        assert len(manifest_files - all_files) == 0

        scan_type = d_modality_scantype.get(modality, modality)
        kwargs = {'manifest': manifest_json}
        if 'diffusion' in scan_type:
            kwargs['bvek_bval_files'] = 'Yes'
        elif 'fMRI' == scan_type:
            kwargs['experiment_id'] = '1942'
        rows += [image03.make_row(sub, ses, scan_type=scan_type, image_description='reconstructed raw data in BIDS format', **kwargs)]

