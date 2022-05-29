import os
import mne
import numpy as np
from ..laura import compute_laura

verbose = 0
spacing = 'ico2'
fs_dir = mne.datasets.fetch_fsaverage(verbose=verbose)
subjects_dir = os.path.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = os.path.join(fs_dir, 'bem', 'fsaverage-trans.fif')
src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')



# Create our own info object, see e.g.:

montage = mne.channels.make_standard_montage("standard_1020")
sfreq = 1000 
info = mne.create_info(montage.ch_names, sfreq, 
    ch_types=['eeg']*len(montage.ch_names), verbose=0)
info.set_montage("standard_1020")

# Create and save Source Model
src = mne.setup_source_space(subject, spacing=spacing, surface='white',
                                    subjects_dir=subjects_dir, add_dist=False,
                                    n_jobs=-1, verbose=verbose)

# Forward Model
fwd = mne.make_forward_solution(info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=-1,
                                verbose=verbose)
fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                    use_cps=True, verbose=verbose)
leadfield = fwd['sol']['data']

def test_laura():
    data = np.random.randn(leadfield.shape[0], 10)
    evoked = mne.EvokedArray(data, info)
    stc = compute_laura(evoked, fwd)
