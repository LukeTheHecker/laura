# **laura**: Local Auto-Regressive Average

This repository contains the source code for the **L**ocal
**AU**to-**R**egressive **A**verage (LAURA) inverse solution as described by de
Peralta Menendez et al. (2001, 2004). The code is based on
[mne-python](https://mne.tools/), a powerful EEG library for
[python](https://python.org/).

Personally, I think this linear inverse solution finds the neural sources
underlying M/EEG measurements with great success and is a valuable option among
other inverse solutions such as *minimum norm estimates* and *(e)LORETA*.

# Dependencies
* [mne](https://mne.tools/stable/index.html)
* [scipy](https://scipy.org/)
* [numpy](https://numpy.org/)  

That's it!
<br/>

# Installation from PyPi
Use [pip](https://pip.pypa.io/en/stable/) to install laura and all its
dependencies from [PyPi](https://pypi.org/project/laura/):

```
pip install laura
```

<br/>

# Quick Start
The following code demonstrates how to use this package:

```
from laura import compute_laura

stc = compute_laura(evoked, forward)
stc.plot()
```
, where evoked is an instance of **mne.Evoked** and forward is an instance of
**mne.Forward**. For further explanation on mne and its objects please refer to
the [mne website](https://mne.tools/).

For a more comprehensive tutorial hop over to [this notebook!](tutorials/tutorial_1.ipynb)

# Feedback
Please leave your feedback and bug reports at lukas_hecker@web.de.

<br/>

# References
Please cite the authors of the LAURA inverse solution appropriately:

[1] Menendez, R. G. D. P., Andino, S. G., Lantz, G., Michel, C. M., &
    Landis, T. (2001). Noninvasive localization of electromagnetic epileptic
    activity. I. Method descriptions and simulations. Brain topography, 14(2),
    131-137.

[2] de Peralta Menendez, R. G., Murray, M. M., Michel, C. M., Martuzzi, R.,
& Andino, S. L. G. (2004). Electrical neuroimaging based on biophysical
constraints. Neuroimage, 21(2), 527-539.

<br/>

I would be happy if you would cite this package, too:

```
LAURA was calculated using the laura python package available at https://github.com/LukeTheHecker/laura.
```

# Limitations
The current implementation is limited to:

* fixed dipole orientations
* time-domain EEG data
  
Feel free to modify the code and start a pull request!

# Troubleshooting
* Having problems with the installation? Check the [package requirements](requirements.txt).