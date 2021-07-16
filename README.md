# ViewHRV

This Repositry contains the open source solution to calculating Accurate and Fast Heart Rate Variability parameters from annotation files. The entire program is written in Python and made available as a web app throuh the usage of Flask. The solution supports the computation of all Time Domain Features, Frequency Domain Features, and Nonelinear Measurements. The web app also supports the MITBIH dataset, where a patient can be selected for whom HRV is later on calculated. The program additionally supports uploading annotation files and calculating HRV for those annotations.

To run the program run the following commands:
```

pip install -r requirements.txt
py WebApp.py

```