"""
File contenente la definizione della classe Main.
Main rappresenta la classe di partenza dell'intera applicazione, definendo il flusso di esecuzione 
e le varie attività che verranno eseguite.
"""
import sys
import traceback

from utils.config_utils import ConfigReader

sys.path.append('../')
sys.path.append('.')
from Execution import Execution
import sys

class Main:

    """
    Main del sistema.
    """
    def main():
        try:
            if len(sys.argv) < 2:
                raise Exception("The name of the dataset is missing. Eg. use dataset1 or dataset2...")
            dataset = sys.argv[1]
            try:
                confReader = ConfigReader(sys.argv)
                conf = confReader.getConf()
                if (dataset not in conf.sections()) and dataset not in ['settings', 'options'] :
                    raise Exception('The name of the dataset is invalid. Eg. use dataset1 or dataset2...')
                dsConf = conf[dataset]
                settings= conf['settings']
                options= conf['options']
            except:
                print("Can't read the configuration file.")
            #generate neighbour images
            size = int(settings['size'])
            if int(options['neighbourImages']) == 1:
                try:
                    getNormalized = int(options['addIndexAndNormalize']) == 1
                    if getNormalized:
                        Execution.normalizeImages(dsConf=dsConf, size=size)
                    Execution.generateNeighbour(dsConf=dsConf, size=size, getNormalized=getNormalized)
                except Exception as exc:
                    traceback.print_exc()
                    print(exc)

            #create dataset
            if int(options['createTrainingSet']) == 1:
                if int(settings['sampled']) == 1:
                    sampling_size = int(settings['sampling_size'])
                    Execution.createTrainingSetSampled(dsConf=dsConf, size=size, settings=settings, sampling_size=sampling_size)
                else:
                    comparison = lambda x, y: x <= y
                    Execution.createDataset(dsConf=dsConf, size=size, comparison=comparison, operation='training')

            if int(options['createTestSet']) == 1:
                comparison = lambda x, y: x > y
                Execution.createDataset(dsConf=dsConf, size=size, comparison=comparison, operation='testing')

            if int(options['createSingleTestSet']) == 1:
                Execution.createSingleTestDataset(dsConf=dsConf, size=size)

            if int(options['trainModel']) == 1:
                optimizeBy = settings.get('optimize_by')
                if not ((optimizeBy == 'f1') or (optimizeBy == 'loss')):
                    raise Exception('The optimization on training fase can only be executed on f1 or loss.')
                try:
                    Execution.trainCNN(confReader=confReader, srcPath=dsConf['datasetPath'], dstPath=dsConf['modelPath'], params=settings, optimizeBy=optimizeBy, size=size)
                except IOError as exc:
                    print(exc)
            if int(options['testModel']) == 1:
                try:
                    Execution.testCNN(confReader=confReader, testPath=dsConf['datasetPath'], modelPath=dsConf['modelPath'], batchSize=int(settings['batchTest']), size=int(settings['size']), numChannels=int(settings['nChannels']))
                except IOError as exc:
                    print(exc)

            if int(options['testSingleImagesModel']) == 1:
                try:
                    Execution.testSingleImagesCNN(confReader=confReader, testPath=dsConf['datasetPath'], modelPath=dsConf['modelPath'], batchSize=int(settings['batchTest']), size=int(settings['size']), numChannels=int(settings['nChannels']))
                except IOError as exc:
                    print(exc)

            if int(options['makeAttentionExplanation']) == 1:
                try:
                    Execution.makeExplanation(confReader=confReader, modelPath=dsConf['modelPath'])
                except IOError as exc:
                    print(exc)

        except Exception as exc:
            print(exc)






if __name__== "__main__":
    Main.main()
