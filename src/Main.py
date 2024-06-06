import sys
sys.path.append('../')
sys.path.append('.')

import traceback
from utils.config_utils import ConfigReader
from Execution import Execution

class Main:

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

            size = int(settings['size'])
            if int(options['neighbourImages']) == 1:
                try:
                    Execution.generateNeighbour(dsConf=dsConf, size=size)
                except Exception as exc:
                    traceback.print_exc()
                    print(exc)

            if int(options['createTrainingSet']) == 1:
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
