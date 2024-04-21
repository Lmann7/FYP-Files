from astropy.io import fits
import pandas as pd
fits_file_path = 'manga_visual_morpho-2.0.1.fits'
hdul = fits.open(fits_file_path)    
table_hdu = hdul[1]  #hdul.info() - Gives info on the HDU's to determine which to use
table_data = table_hdu.data
df = pd.DataFrame(table_data)
hdul.close()

df.to_csv('Mata.csv', index=False)
df1 = pd.read_csv('Mata.csv')
df1 = df1.iloc[:, [1,2,5,6,7]]
df1 = df1[df1.iloc[:, 4] != 1] 
df1.to_csv('MataFlt.csv', index=False)