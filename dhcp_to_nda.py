import os, json, sys, re
import pandas as pd
from pathlib import Path
import argparse
import csv
import numpy as np

from tqdm.auto import tqdm
import pprint
import subprocess

import hashlib
from datetime import datetime
from nda_manifests import *

import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

IMAGE_EXTENSIONS = ['.nii.gz', '.nii']


FILE_BLACKLIST = [
    # TODO: missing files in anat?
    'sub-CC00555XX11_ses-162400_rec-SVR_T1w.json',
    'sub-CC00555XX11_ses-162400_run-06_T2w.json',
    'sub-CC00555XX11_ses-162400_run-06_rec-mc_T2w.json',
    'sub-CC00555XX11_ses-162400_run-06_rec-mcsr_T2w.json',
    'sub-CC00555XX11_ses-162400_run-08_T2w.json',
    'sub-CC00555XX11_ses-162400_run-08_rec-mc_T2w.json',
    'sub-CC00555XX11_ses-162400_run-08_rec-mcsr_T2w.json',
    'sub-CC00555XX11_ses-162400_run-15_T2w.json',
    'sub-CC00555XX11_ses-162400_run-15_rec-mc_T2w.json',
    'sub-CC00555XX11_ses-162400_run-15_rec-mcsr_T2w.json',
    'sub-CC00555XX11_ses-162400_run-17_T1w.json',
    'sub-CC00555XX11_ses-162400_run-17_rec-mc_T1w.json',
    'sub-CC00555XX11_ses-162400_run-17_rec-mcsr_T1w.json',
    'sub-CC00555XX11_ses-162400_run-18_T1w.json',
    'sub-CC00555XX11_ses-162400_run-18_rec-mc_T1w.json',
    'sub-CC00555XX11_ses-162400_run-18_rec-mcsr_T1w.json',
    'sub-CC00555XX11_ses-162400_rec-SVR_T2w.json',
    'sub-CC00555XX11_ses-162400_run-07_acq-MPRAGE_T1w.json',

    'sub-CC00769XX19_ses-6410_rec-SVR_T1w.json',
    'sub-CC00769XX19_ses-6410_run-09_rec-mcsr_T1w.json',
    'sub-CC00769XX19_ses-6410_run-09_rec-mc_T1w.json',
    'sub-CC00769XX19_ses-6410_run-09_T1w.json',
    'sub-CC00769XX19_ses-6410_run-10_rec-mcsr_T1w.json',
    'sub-CC00769XX19_ses-6410_run-10_rec-mc_T1w.json',
    'sub-CC00769XX19_ses-6410_run-10_T1w.json',
    'sub-CC00769XX19_ses-6410_run-11_rec-mcsr_T1w.json',
    'sub-CC00769XX19_ses-6410_run-11_rec-mc_T1w.json',
    'sub-CC00769XX19_ses-6410_run-11_T1w.json',]

DEFAULT_PARAMS = {
    'scanner_manufacturer_pd': "Philips Medical Systems",
    'scanner_type_pd': "Achieva",
    'scanner_software_versions_pd': "3.2.2/3.2.2.0/dHCP",
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
    'image_orientation': 'Axial',
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


def write_csv(dataframe, csv_path, header='"image", "03"\n'):
    with open(csv_path, 'w') as f:
        f.write(header)
    dataframe.to_csv(csv_path, index=False, mode='a', quoting=csv.QUOTE_ALL)


class ManifestSplitter:
    """ split manifest into submanifests with consistent scanner parameters specific to dHCP neo file naming and acquistion"""

    def split(self, modality, manifest, *args, **kwargs):
        if modality == 'anat':
            return self.anat_split(manifest, *args, **kwargs)
        elif modality == 'dwi':
            return self.dwi_split(manifest, *args, **kwargs)
        elif modality == 'func':
            return self.func_split(manifest, *args, **kwargs)
        elif modality == 'B1':
            return self.b1_split(manifest, *args, **kwargs)
        elif modality == 'fmap':
            return self.fmap_split(manifest, *args, **kwargs)

    @staticmethod
    def check_accountedfor(d_submodal_manifest, files, modality, manifest=None):
        accountedfor = []
        notaccountedfor = []
        for v in d_submodal_manifest.values():
            for _v in v:
                if _v in accountedfor:
                    print('WARNING: file accounted for in multiple sub-manifests!', _v['file'])
            accountedfor += v

        notaccountedfor += [p for p in files if p not in accountedfor]

        if notaccountedfor:
            d_submodal_manifest[modality+'-notaccountedfor'] = notaccountedfor
            print('WARNING: change splitting rules? unaccounted for files:\n' + '\n'.join([f['path'] for f in notaccountedfor]),
                  f'full manifest: {manifest}' if manifest is not None else '')

    @staticmethod
    def finalise(d_submodal_manifest, d_submodal_mainimagestem, manifest=None):
        delete_keys = [ ]
        for k in d_submodal_manifest.keys():
            if len(d_submodal_manifest[k]) == 0:
                delete_keys += [k]
                continue

            d_submodal_manifest[k] = {'files': d_submodal_manifest[k]}
            if k not in d_submodal_mainimagestem:
                raise IOError(' '.join(map(str, ['missing file stem for modality', k,
                                                 'in manifest', d_submodal_manifest[k],
                                                 f'manifest: {manifest}' if manifest is not None else '']))
                              )
        for k in delete_keys:
            del d_submodal_manifest[k]
            if k in d_submodal_mainimagestem:
                del d_submodal_mainimagestem[k]


    def fmap_split(self, manifest):
        files = [f for f in manifest['files']]
        d_submodal_manifest = {}
        d_submodal_mainimagestem = {}

        for what in ['magnitude', 'epi']:
            self.get_run_by_pattern(what, files, 'fmap', d_submodal_manifest, d_submodal_mainimagestem)

        self.check_accountedfor(d_submodal_manifest, files, 'fmap', manifest=manifest)
        self.finalise(d_submodal_manifest, d_submodal_mainimagestem, manifest)
        return d_submodal_manifest, d_submodal_mainimagestem


    def b1_split(self, manifest):
        files = [f for f in manifest['files']]
        d_submodal_manifest = {}
        d_submodal_mainimagestem = {}

        for what in ['magnitude']:
            self.get_run_by_pattern(what, files, 'B1', d_submodal_manifest, d_submodal_mainimagestem)

        self.check_accountedfor(d_submodal_manifest, files, 'B1', manifest=manifest)
        self.finalise(d_submodal_manifest, d_submodal_mainimagestem, manifest)
        return d_submodal_manifest, d_submodal_mainimagestem


    def func_split(self, manifest):
        files = [f for f in manifest['files']]
        d_submodal_manifest = {}
        d_submodal_mainimagestem = {}

        for what in ['task-rest_sbref', 'task-rest_bold', 'task-rest_sbref']:
            self.get_run_by_pattern(what, files, 'func', d_submodal_manifest, d_submodal_mainimagestem)

        self.check_accountedfor(d_submodal_manifest, files, 'func', manifest=manifest)
        self.finalise(d_submodal_manifest, d_submodal_mainimagestem, manifest)
        return d_submodal_manifest, d_submodal_mainimagestem

    def dwi_split(self, manifest):
        files = [f for f in manifest['files']]
        d_submodal_manifest = {}
        d_submodal_mainimagestem = {}

        for what in ['sbref', 'dwi']:
            self.get_run_by_pattern(what, files, 'dwi', d_submodal_manifest, d_submodal_mainimagestem)

        # add old recon from release 2, does not have a json but has different image resolution
        # i.e. sub-CC00514XX11_ses-151400_rec-release2_dwi.nii
        rel2dwi = [p for p in files if 'rec-release2_dwi' in p['path']]
        if rel2dwi:
            d_submodal_manifest['dwi-dwiRelease2']  = rel2dwi
            d_submodal_mainimagestem['dwi-dwiRelease2'] = os.path.splitext(d_submodal_manifest['dwi-dwiRelease2'][0]['path'])[0]

        self.check_accountedfor(d_submodal_manifest, files, 'dwi', manifest=manifest)
        self.finalise(d_submodal_manifest, d_submodal_mainimagestem, manifest)
        return d_submodal_manifest, d_submodal_mainimagestem

    @staticmethod
    def get_run_by_pattern(what, files, modality, d_submodal_manifest, d_submodal_mainimagestem):
        # look for json. only use if associated image is found. extract run. find all files from same run
        pattern = re.compile(f'.*(_run-\d\d_){what}.json$')
        imagepattern = re.compile(f'.*(_run-\d\d_){what}(' + '|'.join([re.escape(e) for e in IMAGE_EXTENSIONS]) +')$')
        for p in files:
            if pattern.match(p['path']):
                if not any(imagepattern.match(q['path']) for q in files):
                    print("WARNING: no images found for", p['path'], 'ignoring item')
                    continue
                run = pattern.search(p['path']).group(1)
                newpattern = re.compile('.*'+run+'.*')
                key = f'{modality}-{what}-'+run.rstrip('_').lstrip('_')
                d_submodal_manifest[key] = [f for f in files if newpattern.match(f['path'])]
                d_submodal_mainimagestem[key] = p['path'][:-len('.json')]


    def anat_split(self, manifest):
        files = [f for f in manifest['files']]
        d_submodal_manifest = {}
        d_submodal_mainimagestem = {}

        d_submodal_manifest['anat-T1svr'] = [p for p in files if 'SVR_T1w' in p['path']]
        for f in d_submodal_manifest['anat-T1svr']:
            if f['path'].endswith('SVR_T1w.json'):
                d_submodal_mainimagestem['anat-T1svr'] = f['path'][:-len('.json')]
                break
        d_submodal_manifest['anat-T2svr'] = [p for p in files if 'SVR_T2w' in p['path']]
        for f in d_submodal_manifest['anat-T2svr']:
            if f['path'].endswith('SVR_T2w.json'):
                d_submodal_mainimagestem['anat-T2svr'] = f['path'][:-len('.json')]
                break
        d_submodal_manifest['anat-mprage'] = [p for p in files if 'MPRAGE' in p['path']]
        for f in d_submodal_manifest['anat-mprage']:
            if f['path'].endswith('MPRAGE_T1w.json'):
                d_submodal_mainimagestem['anat-mprage'] = f['path'][:-len('.json')]
                break
        # get individual "runs"
        for what in ['T1w', 'T2w']:
            self.get_run_by_pattern(what, files, 'anat', d_submodal_manifest, d_submodal_mainimagestem)

        self.check_accountedfor(d_submodal_manifest, files, 'anat')
        self.finalise(d_submodal_manifest, d_submodal_mainimagestem)
        return d_submodal_manifest, d_submodal_mainimagestem



class ImageMetadataParser:
    def __init__(self):
        self.image_suffixes = IMAGE_EXTENSIONS
        self.verbose = False

    def info(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def cosine_to_orientation(iop):
        import numpy as np
        """Deduce slicing from cosines

        From bids2nda tool
        From http://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-voxel-to
        -patient-coordinate-system-mapping

        From Section C.7.6.1.1.1 we see that the "positive row axis" is left to
        right, and is the direction of the rows, given by the direction of last
        pixel in the first row from the first pixel in that row. Similarly the
        "positive column axis" is top to bottom and is the direction of the columns,
        given by the direction of the last pixel in the first column from the first
        pixel in that column.

        Let's rephrase: the first three values of "Image Orientation Patient" are
        the direction cosine for the "positive row axis". That is, they express the
        direction change in (x, y, z), in the DICOM patient coordinate system
        (DPCS), as you move along the row. That is, as you move from one column to
        the next. That is, as the column array index changes. Similarly, the second
        triplet of values of "Image Orientation Patient" (img_ornt_pat[3:] in
        Python), are the direction cosine for the "positive column axis", and
        express the direction you move, in the DPCS, as you move from row to row,
        and therefore as the row index changes.

        Parameters
        ----------
        iop: list of float
           Values of the ImageOrientationPatient field

        Returns
        -------
        {'Axial', 'Coronal', 'Sagittal'}
        """
        # Solution based on https://stackoverflow.com/a/45469577
        iop_round = np.round(iop)
        plane = np.cross(iop_round[0:3], iop_round[3:6])
        plane = np.abs(plane)
        if plane[0] == 1:
            return "Sagittal"
        elif plane[1] == 1:
            return "Coronal"
        elif plane[2] == 1:
            return "Axial"
        else:
            raise RuntimeError(
                "Could not deduce the image orientation of %r. 'plane' value is %r"
                % (iop, plane)
            )

    def overwrite_image_parameters(self, modal, filestem, params):
        # fixes wrong or adds missing stuff after get_image_parameters
        if modal.startswith('B1-'):
            params['image_resolution4'] = 4.8
        elif modal.startswith('fmap-magnitude'):
            params['image_resolution4'] = 20.2
        elif modal.endswith('dwiRelease2'):
            params['slice_timing'] = [0, 2.6125, 1.425, 0.2375, 2.85, 1.6625, 0.475, 3.0875, 1.9, 0.7125, 3.325, 2.1375, 0.95, 3.5625, 2.375, 1.1875, 0, 2.6125,
                                      1.425, 0.2375, 2.85, 1.6625, 0.475, 3.0875, 1.9, 0.7125, 3.325, 2.1375, 0.95, 3.5625, 2.375, 1.1875, 0, 2.6125, 1.425,
                                      0.2375, 2.85, 1.6625, 0.475, 3.0875, 1.9, 0.7125, 3.325, 2.1375, 0.95, 3.5625, 2.375, 1.1875, 0, 2.6125, 1.425, 0.2375,
                                        2.85, 1.6625, 0.475, 3.0875, 1.9, 0.7125, 3.325, 2.1375, 0.95, 3.5625, 2.375, 1.1875]
            params['mri_echo_time_pd'] = 0.09
            params['flip_angle'] = 90
            params['mri_repetition_time_pd'] = 3.8
        params['scanner_software_versions_pd'] = DEFAULT_PARAMS['scanner_software_versions_pd']
        if params.get('slice_timing', None) is None:
            params['slice_timing'] = [0]



    def get_image_parameters(self, modal, filestem):
        params = {k:v for k, v in DEFAULT_PARAMS.items()}
        stem = str(filestem)
        suffs = [suf for suf in self.image_suffixes if os.path.isfile(filestem+suf)]
        if not len(suffs) == 1:
            raise IOError('no image found for: ' + stem)
        image_suffix = suffs[0]
        # parse json
        js = {}
        if os.path.isfile(stem+'.json'):
            with open(stem+'.json', 'r') as f:
                js = json.load(f)
            for pk, jk in d_PARAMKEY_JSONKEY.items():
                if jk in js:
                    params[pk] = js[jk]

        # parse image header
        img = nib.load(stem + image_suffix)
        voxel_grid = img.header.get_data_shape()
        voxel_res = img.header.get_zooms()
        units = img.header.get_xyzt_units()

        # TODO replace this hack, properly parse image_orientation --> slice_dimension
        orientation = None
        iop = js.get('ImageOrientationPatient', None)
        if iop is not None:
            orientation = self.cosine_to_orientation(iop)
        if orientation is None:
            if (stem.endswith('T1w') or stem.endswith('T2w')) and 'svr' not in modal.lower() and not 'mprage' in modal: # try to infer using MRtrix3 image strides, does not generalise
                stride = subprocess.check_output(['mrinfo', stem + image_suffix, '-stride']).decode('utf8').split()
                if stride[:3] == ['3', '2', '-1']:
                    orientation = 'Sagittal'
                elif stride[:3] == ['2', '-1', '3']:
                    orientation = 'Axial'
                else:
                    self.warn('warning: stride', stride, 'not recognized, assuming axial for', stem)
                    orientation = 'Axial'
            elif 'mprage' in modal:
                orientation = 'Sagittal'
            else: # here we just assume axial and hope for the best
                orientation = 'Axial'
        # figure out coordinate permutation to RAS coordinate system
        img_ornt = io_orientation(img.affine)
        to_RAS = ornt_transform(img_ornt, axcodes2ornt("RAS"))
        d_niidim2RASdim = {i:j for i, j in zip(range(3), to_RAS[:,0].astype(int))}
        slice_dim = d_niidim2RASdim[{'Axial':2, 'Sagittal':0, 'Coronal':1}.get(orientation)]
        params['image_orientation'] = orientation

        fov = []
        for dim, (size, res) in enumerate(zip(voxel_grid, voxel_res), 1):
            params[f'image_extent{dim}'] = size
            params[f'image_resolution{dim}'] = res
            fov.append(size*res)
            if dim <= 3:
                if units[0] == 'unknown' or units[0] == 'mm':
                    params[f'image_unit{dim}'] = 'Millimeters'
                else:
                    params[f'image_unit{dim}'] = units[0]
        if dim == 4: # TODO generalise to higher dims?
            params[f'image_unit{dim}'] = 'Seconds' # TODO parse
            params[f'image_extent{dim}'] = voxel_grid[3]
            params[f'extent{dim}_type'] = 'volumes'

        params['acquisition_matrix'] = [voxel_grid[d] for d in range(3) if d != slice_dim or 'mprage' in modal]
        params['image_num_dimensions'] = dim
        params['mri_field_of_view_pd'] = [round(fv, 2) for fv in fov[:3]]

        # extract slice thickness from json and NIFTI, apply rules if in disagreement
        if js.get('AcqVoxelSize', None) is not None:
            params['image_slice_thickness'] = js['AcqVoxelSize'][slice_dim] #assumes image_orientation == axial
        if params.get('image_slice_thickness', None) is not None and abs(params['image_slice_thickness'] - voxel_res[slice_dim]) > 1e-4:
            self.info(f"{stem}: using image voxel size {voxel_res[slice_dim]} for slice thickness, not AcqVoxelSize {params['image_slice_thickness']}")
        params['image_slice_thickness'] = voxel_res[slice_dim]

        if modal.startswith('B1'):
            params['mri_repetition_time_pd'] = 0

        # missing = {k:v for k, v in params.items() if v is None}
        # if missing:
        #     print('missing parameters:\n'+pprint.pformat(missing))

        self.overwrite_image_parameters(modal, filestem, params)

        return params


class Image03Gen:
    def __init__(self,
                 d_sub_guid,
                 d_subses_gascan,
                 d_subses_sex,
                 d_subses_interviewdate,
                 image03_definitions,
                 verbose=False
                 ):
        self.d_sub_guid = d_sub_guid
        self.d_subses_gascan = d_subses_gascan
        self.d_subses_sex = d_subses_sex
        self.d_subses_interviewdate = d_subses_interviewdate

        self.df_definitions = pd.read_csv(image03_definitions)

        self.required = self.df_definitions.loc[self.df_definitions.Required == 'Required', :]
        self.conditional = self.df_definitions.loc[self.df_definitions.Required == 'Conditional', :]

        self.Hash = hashlib.sha512
        self.MAX_HASH_PLUS_ONE = 2**(self.Hash().digest_size * 8)

        self.verbose = verbose

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

    @staticmethod
    def info(*args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def make_row(self, sub, ses, **kwargs):
        subses = f"{sub}-{ses}"
        defaults = {'transformation_performed':'No', # if Yes: 'transformation_type':'BIDS2NDA' or similar
                    'image_file_format':'NIFTI',
                    'scan_object':'Live',
                    'image_modality':'MRI'}
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
                if val is None:
                    val = self.d_subses_interviewdate.get(subses, None)
                if val is None: # make up random but reproducible date
                    self.warn(f'making up interview date for {subses}')
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

    TODO: include json and tsv files in top level directory
    """
    def __init__(self, toplevel, manifest_dir, only_subject=None, blacklist=FILE_BLACKLIST):
        self.manifests = {}
        self.file_blacklist = set(blacklist)

        pbar = tqdm([d for d in self.listdirs(toplevel) if d.startswith('sub-')], desc="parsing raw data")
        for sub in pbar:
            if only_subject is not None and sub != only_subject:
                continue

            session_ids = pd.read_csv(toplevel / sub / f'{sub}_sessions.tsv' , delimiter='\t')['session_id'].values.tolist()
            sessions = sorted([f'ses-{ses}' for ses in session_ids])
            session_dirs = sorted([d.name for d in list((toplevel / sub).glob('ses-*')) if os.path.isdir(d)])
            if sessions != session_dirs:
                print("WARNING: sessions " + str(sessions) + " from " + str(toplevel / sub / f'{sub}_sessions.tsv') +
                              " do not match session folders:" + str(session_dirs) + ". Using all.")
                sessions = sorted(set(session_dirs + sessions))

            for ses in sessions:
                for modality in self.listdirs(toplevel / sub / ses):
                    input_dir = toplevel / sub / ses / modality
                    if not os.path.isdir(input_dir):
                        continue
                    manifest_json = manifest_dir / f'{sub}-{ses}_raw{modality}.json'
                    if not os.path.isfile(manifest_json):
                        pbar.set_description(f"Processing {manifest_json}")
                        manifest = self.make_manifest(input_dir)
                        manifest.output_as_file(manifest_dir / f'{sub}-{ses}_raw{modality}.json')
                    with open(manifest_json, 'r') as f:
                        manifest = json.load(f)
                        manifest['files'] = [f for f in manifest['files'] if os.path.split(f['path'])[1] not in self.file_blacklist]
                        if len(manifest['files']) > 0:
                            self.manifests[(sub, ses, modality, input_dir, manifest_json)] = manifest
            # assert not 'CC01116AN11' in sub, {k: v for k, v in self.manifests.items() if k[0]==sub}
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
    parser.add_argument('-id', '--input_dir', help='A directory as input, to create Manifest Files from.')
    parser.add_argument('-out', '--csv_out', help='Write image03.csv to path.')
    parser.add_argument('--manifest_dir', default='manifests', help='output and cache directory for manifest files')
    parser.add_argument('--physlog', action='store_true', help='only write physlogs associated with bold data, ignore other modalities (for post-hock upload)')
    args = parser.parse_args()

    if args.input_dir is None:
        parser.error('No input dir specified')
    print('input dir:', args.input_dir)

    if args.csv_out is None:
        print('writing no output')
    else:
        print('writing csv output to ', args.csv_out)

    print('manifest dir:', args.manifest_dir)
    return args


if __name__ == '__main__':
    pd.options.display.max_rows = 999
    pd.set_option("max_colwidth", 80)

    args = load_args()

    image03_template = 'templates/image03_template.csv'
    image03_definitions = 'templates/image03_definitions.csv'

    # _________________________ parse dHCP metadata _____________________
    url = 'https://raw.githubusercontent.com/BioMedIA/dHCP-release-notes/master/supplementary_files/combined.tsv'
    df_neo = pd.read_csv(url, sep='\t')

    # session GA at scan and subject sex
    d_subses_gascan = {}
    d_subses_sex = {}
    for _, row in df_neo.iterrows():
        sub = 'sub-'+row['participant_id']
        ses = 'ses-'+str(int(row['session_id']))
        d_subses_gascan[sub+'-'+ses] = row['scan_age']
        d_subses_sex[sub+'-'+ses] = {'male':'M', 'female':'F'}.get(row['sex'])

    # subject to guid mapping
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

    # get scan date
    df_scandates = pd.read_csv('dhcp/DHCPNDH1-GUIDS_Randomised_DateOfScanAndSa_DATA_2021-11-19_1316.csv')
    # df_scandates = df_scandates.loc[df_scandates.guid.astype(str).str.startswith('NDA'), :]
    df_scandates = df_scandates[~df_scandates.ses.isna() & ~df_scandates.scan_appt_date.isna()]
    df_scandates['subses'] = 'sub-'+df_scandates.participationid + '-' + 'ses-'+df_scandates.ses.astype(int).astype(str)
    df_scandates['interview_date'] = df_scandates.scan_appt_date.apply(lambda x: f"{x.split('/')[1]}/{x.split('/')[0]}/{x.split('/')[2]}")
    d_subses_interviewdate = {row['subses']:row['interview_date'] for _, row in df_scandates.iterrows()}

    d_modality_scantype = {'anat': 'MR structural (T1, T2)',
                           'B1': 'MR structural (B1 map)',
                           'dwi': 'MR diffusion',
                           'fmap': 'Field Map',
                           'func': 'fMRI'
                           }

    image03 = Image03Gen(d_sub_guid,
                         d_subses_gascan=d_subses_gascan,
                         d_subses_sex=d_subses_sex,
                         d_subses_interviewdate=d_subses_interviewdate,
                         image03_definitions=image03_definitions)

    manifest_dir = Path(args.manifest_dir)
    os.makedirs(manifest_dir, exist_ok=True)

    toplevel = Path(args.input_dir)

    # _________________________ check completeness of cohort metadata _____________________

    data_root_dir = toplevel
    while data_root_dir.name not in ['rel3_derivatives']:
        if data_root_dir == data_root_dir.parent:
            data_root_dir = None
            break
        data_root_dir = data_root_dir.parent

    if data_root_dir is not None:
        print('looking for subjects and sessions in data root dir:', data_root_dir)
        subses_dirlist = sorted(data_root_dir.glob('rel3_rawdata_vol?/sub-*/ses-*'))
        subses4release = set(p.parent.name+'-'+p.name for p in subses_dirlist)

        for subses in sorted(subses4release):
            if subses not in d_subses_interviewdate:
                print('no interview date for', subses)
            if subses not in d_subses_gascan:
                print('no GA scan data for', subses)
            if subses not in d_subses_sex:
                print('no sex for', subses)

        _subses = set(d_subses_gascan).union(set(d_subses_sex)) # not d_subses_interviewdate
        for subses in sorted(_subses - subses4release):
            print('image data for', subses, 'not released?')
        print('sub-ses check done')

    # _________________________ parse dHCP raw images, make modality-specific manifests _____________________

    raw = RawParser(toplevel, manifest_dir)

    # _________________________ parse image metadata, split manifests into coherent submanifests _____________________

    image_parser = ImageMetadataParser()
    manisplit = ManifestSplitter()

    csv_rows = []
    for (sub, ses, modality, input_dir, manifest_json), manifest in tqdm(raw.manifests.items(), desc="parsing image metadata data"):
        # check if all files are accounted for
        all_files = []
        for path, subdirs, files in os.walk(input_dir):
            for name in files:
                f = os.path.join(path, name)
                if os.path.isfile(f):
                    all_files.append(Path(f).absolute())
        manifest_files = set([Path(f['path']).absolute() for f in manifest['files']])
        all_files = set([f for f in all_files if f.name not in set(FILE_BLACKLIST)])
        assert len(all_files - manifest_files) == 0
        assert len(manifest_files - all_files) == 0

        # subject and session and other general metadata
        scan_type = d_modality_scantype.get(modality, modality)
        kwargs = {'manifest': manifest_json}
        if 'diffusion' in scan_type:
            kwargs['bvek_bval_files'] = 'Yes'
        elif 'fMRI' == scan_type:
            kwargs['experiment_id'] = '1942'

        row = image03.make_row(sub, ses, scan_type=scan_type, image_description='reconstructed raw data in BIDS format', **kwargs)

        # split manifest by scan, parse main image in each scan to fill in image properties fields
        d_submodal_manifest, d_submodal_mainimagestem = manisplit.split(modality, manifest)
        # pprint.pprint({k:[p['path'] for p in v['files']] for k, v in d_submodal_manifest.items()})
        for modal, stem in d_submodal_mainimagestem.items():
            submanifest = d_submodal_manifest[modal]

            # write submanifest
            json_path = manifest_dir / f'submanifest_{sub}_{ses}_raw{modal}.json'
            with open(json_path, 'w') as f:
                f.write(json.dumps(submanifest))

            imdata = {'_modality': modal, '_main_image': stem}
            imdata.update(image_parser.get_image_parameters(modal, stem))

            # hack to add physlogs to bold fMRI data via separate manifests
            if modal.startswith('func') and 'bold' in modal:
                # if sub == 'sub-CC00063AN06' and ses == 'ses-15102':
                #     print(sub, ses, stem, submanifest)

                physlog = Path(args.input_dir).parent / 'rel3_sourcedata' / sub / ses / 'func'
                fname = Path(stem).name.replace('task-rest_bold', 'task-rest_physio.log')
                physlog = physlog / fname
                # .glob(f'{sub}_{ses}_run-*_task-rest_physio.log'))
                if not os.path.isfile(physlog):
                    print("WARNING: manifest missing for %s" % row)
                    continue

                physlog_json_path = manifest_dir / f'submanifest_{sub}_{ses}_raw{fname}.json'

                if True or not os.path.isfile(physlog_json_path):
                    manifest = Manifest()
                    manifest.files.append(ManifestRecord(str(physlog)))
                    manifest.output_as_file(physlog_json_path)

                imdata.update(row)
                imdata['manifest'] = str(physlog_json_path)
                csv_rows += [imdata]

            if args.physlog: # hack to add physlog: ignore everything else as already uploaded
                continue

                #     physlog = str(physlog[0].relative_to(Path(args.input_dir).parent))
                #     row['data_file2'] = physlog
                # else:
                #     continue

            imdata.update(row)
            imdata['manifest'] = str(json_path)
            csv_rows += [imdata]


    # _________________________ write image03 _____________________
    if args.csv_out:
        df_image03 = pd.DataFrame(csv_rows)
        # remove debug columns
        df_image03_out = df_image03.reindex([c for c in sorted(df_image03.columns) if not c.startswith('_')], axis=1)
        # prevent error image_extent4: The field provided was not an integer.
        df_image03_out.image_extent4 = df_image03_out.image_extent4.apply(lambda x: str(int(x)) if isinstance(x, float) and not np.isnan(x) else x)
        write_csv(df_image03_out, args.csv_out)

        df_image03_debug = df_image03.reindex([c for c in sorted(df_image03.columns)], axis=1)
        df_image03_debug.to_csv(str(args.csv_out)[:-len('.csv')]+'_debug.csv', index=False)


