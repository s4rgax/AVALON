import rasterio
import numpy as np
from src.preprocessing.ImageException import ImageException
from tiler import Tiler
import pickle
import os


class Image:

    """
    Constructor method for Image class
    used to perform operations on Sentinel Images
        
    :param path: source path of the image
    """
    def __init__(self, path: str = None) -> None:
        try:
            with rasterio.open(path, "r") as img:
               self.id : [str] = os.path.splitext(os.path.basename(img.name))[0]
               self.height : [int] = img.height
               self.width : [int] = img.width
               self.bands : [int] = img.count
               self.profile = img.profile
               self.pixel = img.read()
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
    Returns the ID of the source image
    
    :return: image ID
    """
    def getID(self) -> str:
        return self.id
    

    """
    Set the ID of the source image
    
    :param id: numeric ID of the source image
    """
    def setID(self, id: str) -> None:
        self.id = id
    

    """
    Return the pixel in x,y coordinates

    :param y: y coordinate for the extracted pixel
    :param x: x coordinate for the extracted pixel
    :return: values for pixel in x,y coordinates
    """
    def getPixel(self, y: int, x: int) -> 'np.array[float]':
        return self.pixel[:, y, x]
    

    """
    Set the given values on the pixel in x,y coordinates

    :param y: y coordinate for the selected pixel
    :param x: x coordinate for the selected pixel
    :param px: values to insert for pixel in coordinates x,y
    """
    def setPixel(self, y: int, x: int, px: 'np.array[float]') -> None:
        self.pixel[:, y, x] = px


    """
    Returns the whole image values

    :return: all pixel values of the image
    """
    def getAllPixel(self) -> 'Image.pixel':
        return self.pixel
    

    """
    Set the entire set of values for an iamge

    :param setPx: set of pixel values
    """
    def setAllPixel(self, setPx: 'np.array[float][float][float]') -> None:
        self.pixel = setPx


    """
    Returns the width of the image

    :return: width of the image
    """
    def getWidth(self) -> int:
        return self.width


    """
    Set the width for class Image

    :param: width value
    """
    def setWidth(self, w: int) -> None:
        self.width = w
    

    """
    Returns the height of the image

    :return: height of the image
    """
    def getHeight(self) -> int:
        return self.height
    

    """
    Set the height for class Image

    :param: height value
    """
    def setHeight(self, h: int) -> None:
        self.height = h
    

    """
    Returns the number of channels for the Image

    :return: num channels
    """
    def getBands(self) -> int:
        return self.bands


    """
    Set the number of channels for the Image

    :param b: valore intero da impostare come numero di canali dell'immagine.
    """
    def setBands(self, b: int) -> int:
        self.bands = b


    """
    Returns profile and metadata of the image
    Restituisce il profilo dell'immagine contenente i metadati associati ad essa.

    :return: profile and metadata of image
    """
    def getProfile(self) -> 'rasterio.profiles.Profile':
        return self.profile
    

    """
    Set profile and metadata of the image.

    :param profile: profile and metadata of image.
    """
    def setProfile(self, profile: 'rasterio.profiles.Profile') -> None:
        self.profile = profile

    """
    Extracts neighborhood images of size equal to 'size' for each pixel in the image
    and saves them to the folder specified in input.

    :param size: size of Neighbour images to be extracted from the current image.
    :param path: path to the directory in which to save the extracted images.
    :savePickle: value indicating how the extracted images are saved.
               With 1 the images are saved in pickle format, otherwise in 0 format.
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
    Allows the source image to be enlarged by exploiting the size of the neighborhood images to be extracted.
    It is used to improve the efficiency of neighborhood extraction.

    :param size: size to be considered for padding; must be an odd value.
    :return padImg: original image extended with dummy values (NaN) along edges.
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
    Saves the image to the destination specified in input.

    :param path: path to the directory in which to save the image.
    """
    def save(self, path: str) -> None:
        try:
            dstPath = f"{path}{self.getID()}.TIF"
            with rasterio.open(dstPath, "w", **self.profile) as dst:
                dst.write(self.getAllPixel())
        except:
            raise ImageException("Error during saving TIF.")


    """
    Saves the image in Pickle format to the destination specified in input.

    :param path: path to the directory in which to save the image.
    """
    def saveAsPickle(self, path: str) -> None:
        try:
            with open(f"{path}{self.id}.pickle", "wb") as file:
                pickle.dump(self.getAllPixel(), file)
        except Exception as e:
            print("Error during saving pickle.", e)
            # raise ImageException("Error during saving pickle.", e)


    """
    Print method.
    """
    def __str__(self) -> None:
        msg = f"ID: {self.getID()}\nH: {self.getHeight()}\nL: {self.getWidth()}\nPixel: \n"
        for band in range(self.pixel.shape[0]):
            msg += f"Band: {band}\n{self.pixel[band, :, :]}\n"
        return msg


    """
    Method for comparing instances of the class.

    :param __value: image to compare with current image.'
    """
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Image):
            return False
        if  not self.getHeight() == __value.getHeight():
                return False
        if not self.getWidth() == __value.getWidth():
            return False
        return np.array_equal(self.getAllPixel(), __value.getAllPixel())