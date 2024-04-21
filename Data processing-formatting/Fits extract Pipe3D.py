from astropy.io import fits
import pandas as pd

fits_file_path = 'SDSS17Pipe3D_v3_1_1.fits'
hdul = fits.open(fits_file_path)
table_hdu = hdul[1]
table_data = table_hdu.data
hdul.close()

df = pd.DataFrame(table_data)
df.to_csv('PIPE3D.csv', index=False)

df1 = pd.read_csv('PIPE3D.csv')
# List of strings to exclude
exclude_strings = ['objra', 'objdec', 'P(', 'best_type', 'error']

df1 = df1[[col for col in df.columns if not any(ex_str in col or col.startswith('e_') or col.startswith('a_') for ex_str in exclude_strings)]]
df1.to_csv('PIPE3DFlt.csv', index=False)