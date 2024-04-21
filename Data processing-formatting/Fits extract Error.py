from astropy.io import fits
import pandas as pd
fits_file_path = 'manga_visual_morpho-2.0.1.fits'
hdul = fits.open(fits_file_path)    
table_hdu = hdul[1]  #hdul.info() - Gives info on the HDU's to determine which to use
table_data = table_hdu.data
df = pd.DataFrame(table_data)
hdul.close()

#df.to_csv('Mata.csv', index=False)
#df = pd.read_csv('Mata.csv')
df = df.iloc[:, [1,2,4,5,6,7]]
df = df[df.iloc[:, 5] != 1] 
df.to_csv('MataFlt.csv', index=False)