import os, struct

path = "/home/hamit/Softwares/Dynamic3DGaussians/output/2025-08-06_16-09-24_3412x2500_combin2_all_test1/2025-08-06_16-09-24_3412x2500_combin2_all/params.npz"  # put your filename here


def fixBadZipfile(zipFile):
 f = open(zipFile, 'r+b')
 data = f.read()
 pos = data.find(b'\x50\x4b\x05\x06') # End of central directory signature
 if (pos > 0):
     self._log("Trancating file at location " + str(pos + 22)+ ".")
     f.seek(pos + 22)   # size of 'ZIP end of central directory record'
     f.truncate()
     f.close()
 else:
     raise ValueError("error")
fixBadZipfile(path)
