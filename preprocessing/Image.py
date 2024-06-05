"""
File contenente la definizione della classe Image.
Image rappresenta una determinata immagine, conoscendo le sue caratteristiche e consentendo le
varie operazioni di manipolazione dell'immagine.
"""
import rasterio
import numpy as np
# from preprocessing.ImageException import ImageException
from src.preprocessing.ImageException import ImageException
from tiler import Tiler
import pickle
import os


class Image:

    """
    Metodo costruttore dell classe.
        
    :param path: Percorso sorgente dell'immagine 
    """
    def __init__(self, path: str = None) -> None:
        try:
            with rasterio.open(path, "r") as img:
               self.id : [str] = os.path.splitext(os.path.basename(img.name))[0]   #nome immagine
               self.height : [int] = img.height                                    #altezza immagine
               self.width : [int] = img.width                                      #lunghezza immagine
               self.bands : [int] = img.count                                      #numero delle bande dell'immagine
               self.profile = img.profile                                          #metadati dell'immagine
               self.pixel = img.read()                                             #pixel dell'immagine    [Depth, Height, Width]
               self.transform = img.transform
               self.crs = img.crs
               np.nan_to_num(self.pixel, copy = False)
        except:
            try:
                with open(path, "rb") as img:
                    self.id : [str] = os.path.splitext(os.path.basename(path))[0]
                    self.pixel = pickle.load(img)
                    self.height : [int] = self.pixel.shape[1]
                    self.width : [int] = self.pixel.shape[2]
                    self.bands : [int] = self.pixel.shape[0]
                    self.profile = None
            except:
                self.id = None
                self.profile = None
                self.height = None
                self.width = None
                self.bands = None
                self.pixel = None
    

    """
    Restituisce l'ID dell'immagine.
    
    :return: ID associato all'immagine.
    """
    def getID(self) -> str:
        return self.id
    

    """
    Imposta l'ID passato in input come ID dell'immagine.

    :param id: ID da impostare come ID dell'immagine.

    """
    def setID(self, id: str) -> None:
        self.id = id
    

    """
    Restituisce il pixel dell'immagine in posizione (x, y).

    :param y: coordinata y del pixel da estrarre.
    :param x: coordinata x del pixel da estrarre.
    :return: pixel dell'immagine in posizione (x, y).
    """
    def getPixel(self, y: int, x: int) -> 'np.array[float]':
        return self.pixel[:, y, x]
    

    """
    Inserisce nella posizione (x, y) dell'immagine il pixel passato in input.

    :param y: coordinata y della posizione nell'immagine in cui inserire il pixel.
    :param x: coordinata x della posizione nell'immagine in cui inserire il pixel.
    :param px: pixel da inserire nell'immagine in posizione (x, y).
    """
    def setPixel(self, y: int, x: int, px: 'np.array[float]') -> None:
        self.pixel[:, y, x] = px


    """
    Restituisce l'intero set di pixel dell'immagine.

    :return: tutti i pixel dell'immagine.
    """
    def getAllPixel(self) -> 'Image.pixel':
        return self.pixel
    

    """
    Assegna all'immagine un intero set di pixel complessivamente in 
    un'unica operazione.

    :param setPx: insieme di pixel da assegnare all'immagine.
    """
    def setAllPixel(self, setPx: 'np.array[float][float][float]') -> None:
        self.pixel = setPx


    """
    Restituisce la larghezza dell'immagine.

    :return: larghezza dell'immagine.
    """
    def getWidth(self) -> int:
        return self.width


    """
    Assegna all'immagine la larghezza passata in input.

    :param: valore intero da impostare come larghezza dell'immagine.
    """
    def setWidth(self, w: int) -> None:
        self.width = w
    

    """
    Restituisce l'altezza dell'immagine.

    :return: altezza dell'immagine.
    """
    def getHeight(self) -> int:
        return self.height
    

    """
    Assegna all'immagine l'altezza passata in input.

    :param h: valore intero da impostare come larghezza dell'immagine.
    """
    def setHeight(self, h: int) -> None:
        self.height = h
    

    """
    Restituisce il numero di bande dell'immagine.

    :return: numero dei canali dell'immagine.
    """
    def getBands(self) -> int:
        return self.bands


    """
    Restituisce il numero di bande dell'immagine.

    :param b: valore intero da impostare come numero di canali dell'immagine.
    """
    def setBands(self, b: int) -> int:
        self.bands = b


    """
    Restituisce il profilo dell'immagine contenente i metadati associati ad essa.

    :return: metadati dell'immagine.
    """
    def getProfile(self) -> 'rasterio.profiles.Profile':
        return self.profile
    

    """
    Imposta il profilo dell'immagine contenente i metadati corrispondenti.

    :param profile: metadati da associare all'immagine. 
    """
    def setProfile(self, profile: 'rasterio.profiles.Profile') -> None:
        self.profile = profile

    def getCrs(self):
        return self.crs

    def getTransform(self):
        return self.transform

    """
    Estrae le immagini del vicinato di dimensione pari a 'size' per ogni pixel dell'immagine
    e le salva nella cartella specificata in input.

    :param size: dimensione delle immagine Neighbour da estrarre dall'immagine corrente.
    :param path: path della directory in cui salvare le immagini estratte.
    :savePickle: valore che indica la modalità di salvataggio delle immagini estratte.
               Con 1 le immagini vengono salvate in formato pickle, altrimenti in formato 0.
    """
    def extractNeighborhood(self, size: int, basepath: str) -> None:
        #setup tiling parameters
        tiler = Tiler(
            data_shape = (self.getBands(), self.getHeight(), self.getWidth()),
            tile_shape = (self.getBands(), size, size),
            overlap = size - 1,
            channel_dimension = 0,
        )

        for tileId, tile in tiler.iterate(self.getAllPixel()):
            tile = np.where(np.isnan(tile),
                            np.nanmean(tile,
                                       axis = (1, 2)
                                       )[:, np.newaxis, np.newaxis],
                            tile)
            #creating neighbourImage
            nImage = Image()
            nImage.setID(f"({tileId % (self.getWidth() - size + 1)},{tileId // (self.getWidth() - size + 1)})_{self.id}_{size}")
            nImage.setAllPixel(tile)
            #saving neighbour images

            path = f"{basepath}Neighbour_{size}/"
            os.makedirs(path, exist_ok = True)

            nImage.saveAsPickle(path)

    """
    Permette di allargare l'immagine di partenza sfruttando la dimensione delle immagini
    del vicinato da estarre.
    Serve per migliorare l'efficienza dell'estrazione del vicinato.

    :param size: dimensione da considerare per il padding; deve essere un valore dispari.
    :return padImg: immagine originale allargata con valori fittizi (NaN) lungo i bordi.
    """
    def padding(self, size: int) -> 'Image':
        padImg = Image()
        padImg.setID(self.getID())
        padImg.setHeight(self.getHeight() + size - 1)
        padImg.setWidth(self.getWidth() + size - 1)
        padImg.setBands(self.getBands())
        padImg.setProfile(self.getProfile())
        padImg.pixel = np.full(shape = (padImg.getBands(), padImg.getHeight(), padImg.getWidth()),
                                fill_value = np.nan)
        bound = int(size / 2)
        padImg.pixel[:, bound : padImg.getHeight() - bound, bound : padImg.getWidth() - bound] = self.pixel[:]
        
        return padImg
    

    """
    Salva l'immagine nella destinazione specificata in input.

    :param path: path della directory in cui salvare l'immagine.
    """
    def save(self, path: str) -> None:
        try:
            dstPath = f"{path}{self.getID()}.TIF"
            with rasterio.open(dstPath, "w", **self.profile) as dst:
                dst.write(self.getAllPixel())
        except:
            raise ImageException("Error during saving TIF.")


    """
    Salva l'immagine in formato Pickle nella destinazione specificata in input.

    :param path: path della directory in cui salvare l'immagine.
    """
    def saveAsPickle(self, path: str) -> None:
        try:
            with open(f"{path}{self.id}.pickle", "wb") as file:
                pickle.dump(self.getAllPixel(), file)
        except Exception as e:
            print("Error during saving pickle.", e)
            # raise ImageException("Error during saving pickle.", e)


    """
    Metodo di stampa.
    """
    def __str__(self) -> None:
        msg = f"ID: {self.getID()}\nH: {self.getHeight()}\nL: {self.getWidth()}\nPixel: \n"
        for band in range(self.pixel.shape[0]):
            msg += f"Band: {band}\n{self.pixel[band, :, :]}\n"
        return msg


    """
    Metodo di confronto tra istanze della classe.

    :param __value: immagine da confrontare con l'immagine corrente.'
    """
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Image):
            return False
        if  not self.getHeight() == __value.getHeight():
                return False
        if not self.getWidth() == __value.getWidth():
            return False
        return np.array_equal(self.getAllPixel(), __value.getAllPixel())

    def print_img_infos(self):
        print(f'[{self.id}] {self.height}x{self.width}x{self.bands}')

    """def add_vegetation_features(self, feature_matrix):
        return np.column_stack((feature_matrix, NMDI(feature_matrix), MCARI(feature_matrix), NGDRI(feature_matrix)))"""

    """def add_scl_feature(self, feature_matrix):
        feature_matrix[:,band_dictionary['B013']] = SCL(feature_matrix)
        return feature_matrix"""

band_dictionary = {'B01': 0, 'B02':1, 'B03' :2 , 'B04': 3, 'B05': 4,'B06' :5, 'B07': 6, 'B08':7, 'B08A':8,
                   'B09': 9, 'B011':10, 'B012':11, 'B013':12}


def add_vegetation_features(feature_matrix):
    return np.column_stack((feature_matrix, NMDI(feature_matrix), MCARI(feature_matrix), NGDRI(feature_matrix)))


def add_scl_feature(feature_matrix):
    feature_matrix[:, band_dictionary['B013']] = SCL(feature_matrix)
    return feature_matrix

def NGDRI(XTrain): #(B03 - B04) / (B03 + B04);
    B03=XTrain[:,band_dictionary['B03']]
    B04=XTrain[:,band_dictionary['B04']]
    d=(B03 + B04)
    NGDRI=np.where(d == 0, 0, ((B03 - B04)/d))
    return NGDRI


def NMDI(XTrain): #NMDI=(NIR−(SWIR1−SWIR2))/(NIR+(SWIR1+SWIR2))
    B08 = XTrain[:, band_dictionary['B08']]
    B011 = XTrain[:, band_dictionary['B011']]
    B012 = XTrain[:, band_dictionary['B012']]
    d=(B08+B011+B012)
    NMDI = np.where(d == 0, 0, ((B08-(B011-B012)) / d))
    return NMDI


def MCARI (XTrain): # ((B05 - B04) - 0.2 * (B05 - B03)) * (B05 / B04)
    B03 = XTrain[:, band_dictionary['B03']]
    B04 = XTrain[:, band_dictionary['B04']]
    B05 = XTrain[:, band_dictionary['B05']]
    MCARI=np.where((B04==0), 0,((B05 - B04) - 0.2 * (B05 - B03)) * (B05 / B04))
    return MCARI

def SCL(bands_matrix):
    VEGETATION_VALUE = 4.
    SCL = bands_matrix[:, band_dictionary['B013']]
    SCL = np.where(SCL == VEGETATION_VALUE, 1, 0)
    return SCL

def createGeoTiff(array_npy, outputFileName, transform):
    n_bands, height, width = array_npy.shape
    # Define the metadata for the GeoTIFF
    metadata = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': n_bands,  # Number of bands
        'dtype': 'float32',  # Data type of the array
        'crs': 'epsg:4326',
        'transform': transform
    }
    # Create the GeoTIFF file and write the data
    with rasterio.open(outputFileName, 'w', **metadata) as f:
        f.write(array_npy)  # Write the data to band 1
        """for band in range(n_bands):
            f.write(array_npy[band, :, :], band + 1)  # Write the data for each band"""