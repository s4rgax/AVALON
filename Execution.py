"""
File contenente la definizione e l'implementazione della classe Execution, la quale
contiene dei metodi di classi rappresentanti le varie operazioni eseguibili dal sistema.
Si occupa pertanto dell'implementazione e dell'esecuzione delle varie operazioni del sistema.
"""
import json
import os
import random
import traceback
from copy import copy

import cv2
import joblib
import tifffile
from sklearn.preprocessing import MinMaxScaler

# from preprocessing.Image import Image
from src.preprocessing.Image import Image, createGeoTiff
from tqdm import tqdm
import pickle
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from cnn.Cnn import CNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from cnn.EarlyStopper import EarlyStopper
import csv
import pandas as pd
import time
from torchvision.ops import sigmoid_focal_loss
from typing import Callable
from src.preprocessing.Image import add_scl_feature, add_vegetation_features
# from pytorch_model_summary import summary
from PIL import Image as ImagePIL

from src.utils.explainability import save_from_scaled_attention_map, computeAttentionScene, computePixelBoundary, \
    extract_neigh_from_predictions, save_from_scaled_attention_map_with_label, newComputeAttentionScene


class Execution:

    """
    Metodo che genera tutte le immagini Neighbour di dimensione 'size' a partire dalle immagini presenti nella directory
    'sourceDir' e le salva nella directory 'destinationDir' con la modalità 'savePickle'.

    :param souceDir: path della directory sorgente da cui estrarre le immagini da cui estrarre le immagini Neighbour.
    :param size: dimensione delle immagini Neighbour da generare.
    :param destinationDir: path della directory di destinazioni in cui salvare le immagini generate.
    :param savePickle: valore che specifica la modalità di salvataggio delle immagini generate.
                       Con 1 le immagini vengono salvate in formato pickle, con 0 in formato Tif.
    """
    def generateNeighbour(dsConf, size: int, getNormalized= False) -> None:
        sourceDir= dsConf['normImagesPath'] if getNormalized else dsConf['imagesPath']
        destinationDir = dsConf['neighbourPath']
        os.makedirs(destinationDir, exist_ok = True)
        print(f'Getting images from {sourceDir}')
        for fileName in tqdm(os.listdir(sourceDir), desc = "Extracing Neighborhood"):
            if fileName.endswith('tif'):
                image = Image(f"{sourceDir}{fileName}")
                # image.print_img_infos()
                padImage = image.padding(size)
                padImage.extractNeighborhood(size, destinationDir)
        print("Neighbour Images generated.")


    """
    Metodo che permette di creare il dataset (trainingSet e/o testSet) da utilizzare per la CNN e lo salva nella directory specificata.

    :param dsConf: parametri specifici del dataset a cui si sta operando e che definisce i vari path e impostazioni corrispondenti.
    :param size: dimensione delle immagini neighbour che vengono considerate per la creazione del dataset.
    :param comparison: funzione di confronto che serve per distinguere il set di immagini da considerare nella creazione del dataset.
                       Dipende se si sta creando il trainingSet e il testSet.
    :param operation: nome del tipo di dataset (trainingSet o testSet) che stiamo costruendo. Definisce il nome da utilizzare nella memorizzazione dei file.
                      NB: usare training o testing per garantire compatibilità con gli altri metodi. 
    """

    def createDataset(dsConf: dict, size: int, comparison: Callable, operation: str ) -> None:
        labelsSet = np.array([])
        dataSet = []
        imagesDir = f"{dsConf['neighbourPath']}Neighbour_{size}/"
        os.makedirs(imagesDir, exist_ok=True)
        os.makedirs(dsConf['datasetPath'], exist_ok=True)
        dataset_filepath = os.path.join(dsConf['datasetPath'],f"{operation}Data_{size}.pickle")
        labels_filepath = os.path.join(dsConf['datasetPath'],f"{operation}Labels_{size}.pickle")
        if os.path.exists(dataset_filepath):
            os.remove(dataset_filepath)
        if os.path.exists(labels_filepath):
            os.remove(labels_filepath)
        for fileName in tqdm(filter(lambda x: comparison(int(x.split('_')[1].split('.')[0]), int(dsConf['trainBound'])),
                               sorted(os.listdir(dsConf['maskPath']),
                                key = lambda x: int(x.split('_')[1].split('.')[0]))), desc = f'Creating {operation} set'):
            with rasterio.open(f"{dsConf['maskPath']}{fileName}", "r") as mask:
                try:
                    nMask = os.path.basename(mask.name).split('_')[1].split('.')[0]
                    labels = mask.read().reshape(-1)
                    labels[labels == 255] = 1.
                    for index in range(len(labels)):
                        row = index // mask.width
                        column = index % mask.width
                        with open(f"{imagesDir}({column},{row})_geojson_{nMask}_{size}.pickle", "rb") as img:
                            tile = pickle.load(img)
                            dataSet.append(tile)
                    labelsSet = np.append(labelsSet, labels)
                except Exception as exc:
                    traceback.print_exc()
                    raise exc
        try:
            print(f'Saving in{dsConf["datasetPath"]}')
            with open(dataset_filepath, "wb") as file:
                pickle.dump(np.array(dataSet), file)
            with open(labels_filepath, "wb") as file:
                pickle.dump(labelsSet, file)
            print("Dataset created and saved.")
        except:
             print("Error during saving.")

    def createSingleTestDataset(dsConf: dict, size: int) -> None:
        comparison = lambda x, y: x > y
        operation = 'single_test'
        imagesDir = f"{dsConf['neighbourPath']}Neighbour_{size}/"
        os.makedirs(imagesDir, exist_ok=True)
        single_data_path = os.path.join(dsConf['datasetPath'], 'single_test', 'data')
        single_labels_path = os.path.join(dsConf['datasetPath'], 'single_test', 'mask')
        os.makedirs(single_data_path, exist_ok=True)
        os.makedirs(single_labels_path, exist_ok=True)
        for fileName in tqdm(filter(lambda x: comparison(int(x.split('_')[1].split('.')[0]), int(dsConf['trainBound'])),
                               sorted(os.listdir(dsConf['maskPath']),
                                key = lambda x: int(x.split('_')[1].split('.')[0]))), desc = f'Creating {operation} set'):
            labelsSet = np.array([])
            dataSet = []
            id = int(fileName.split('_')[1].split('.')[0])
            with rasterio.open(f"{dsConf['maskPath']}{fileName}", "r") as mask:
                try:
                    nMask = os.path.basename(mask.name).split('_')[1].split('.')[0]
                    labels = mask.read().reshape(-1)
                    labels[labels == 255] = 1.
                    width = mask.width
                    height = mask.height
                    for index in range(len(labels)):
                        row = index // mask.width
                        column = index % mask.width
                        with open(f"{imagesDir}({column},{row})_geojson_{nMask}_{size}.pickle", "rb") as img:
                            tile = pickle.load(img)
                            dataSet.append(tile)
                    labelsSet = np.append(labelsSet, labels)

                    data_filepath = os.path.join(single_data_path, f"{id}_{width}x{height}.pickle")
                    labels_filepath = os.path.join(single_labels_path, f"{id}_{width}x{height}.pickle")
                    if os.path.exists(data_filepath):
                        os.remove(data_filepath)
                    if os.path.exists(labels_filepath):
                        os.remove(labels_filepath)
                    try:
                        with open(data_filepath, "wb") as file:
                            pickle.dump(np.array(dataSet), file)
                        with open(labels_filepath, "wb") as file:
                            pickle.dump(labelsSet, file)
                    except Exception as e:
                        print(f"Error during saving single files for id {id}: {e}")

                except Exception as exc:
                    traceback.print_exc()
                    raise exc
    """
    Metodo che permette di creare il trainingSet campionato da utilizzare per la CNN e lo salva nella directory specificata.

    :param dsConf: parametri specifici del dataset a cui si sta operando e che definisce i vari path e impostazioni corrispondenti.
    :param size: dimensione delle immagini neighbour che vengono considerate per la creazione del trainingSet.
    """
    def createTrainingSetSampled(dsConf: dict, settings: dict, size: int, sampling_size: int):
        labelsSet = np.array([])
        dataSet = []
        imagesDir = f"{dsConf['neighbourPath']}Neighbour_{size}/"
        os.makedirs(imagesDir, exist_ok=True)
        os.makedirs(dsConf['datasetPath'], exist_ok=True)
        print(f'Sampling every {sampling_size} pixel')
        for fileName in tqdm(filter(lambda x: int(x.split('_')[1].split('.')[0]) <= int(dsConf['trainBound']),
                               sorted(os.listdir(dsConf['maskPath']),
                                key = lambda x: int(x.split('_')[1].split('.')[0]))), desc = 'Creating Dataset'):
            with rasterio.open(f"{dsConf['maskPath']}{fileName}", "r") as mask:
                nMask = mask.name.split('_')[1].split('.')[0]
                labels = mask.read().reshape(-1)
                labels[labels == 255] = 1.
                positions = np.concatenate([np.where(labels == 0)[0][::sampling_size], np.where(labels == 1)[0]])
                positions = np.sort(positions)
                for index in positions:
                    row = index // mask.width
                    column = index % mask.width
                    with open(f"{imagesDir}({column},{row})_geojson_{nMask}_{size}.pickle", "rb") as img:
                        tile = pickle.load(img)
                        dataSet.append(tile)
                boolMask = np.zeros(labels.shape, dtype = bool) #filtro delle labels
                boolMask[positions] = True
                labels = labels[boolMask]
                labelsSet = np.append(labelsSet, labels)
        try:
            with open(f"{dsConf['datasetPath']}trainingDataSampled_{size}.pickle", "wb") as file:
                pickle.dump(np.array(dataSet), file)
            with open(f"{dsConf['datasetPath']}trainingLabelsSampled_{size}.pickle", "wb") as file:
                pickle.dump(labelsSet, file)
            print("Dataset created and saved.")
        except:
             print("Error during saving.")

    """
    Metodo che si occupa della creazione di una CNN con successivo addestramento e testing sul dataset contenuto
    nel percorso sorgente passato in input utilizzando degli iperparametri ottimizzati.
    Oltre al modello migliore, viene salvato anche un file csv contenente i parametri utilizzati e i risultati
    ottenuti in ogni tentativo.

    :param srcPath: path della directory sorgente contenente il dataset su cui addestrare e validare il modello.
    :param params:  parametri che vengono passati durante la definizione e l'addestramente del modello CNN.
    """
    def trainCNN(confReader:any, srcPath: str, dstPath: str, params: dict, optimizeBy:str, size: int) -> None:
        global sampled
        sampled = ''
        if int(params['sampled']) == 1:
            sampled = 'Sampled'
        #getting dataset
        print("Getting dataset...", end = " ")
        try:
            with open(f"{srcPath}trainingData{sampled}_{params['size']}.pickle", "rb") as file:
                dataset = pickle.load(file)
            with open(f"{srcPath}trainingLabels{sampled}_{params['size']}.pickle", "rb") as file:
                labels = pickle.load(file)
            with open(f"{srcPath}testingData_{params['size']}.pickle", "rb") as file:       #TO DELETE?
                testingData = pickle.load(file)
            with open(f"{srcPath}testingLabels_{params['size']}.pickle", "rb") as file:     #TO DELETE?
                testingLabel = pickle.load(file)
        except:
            raise IOError("Can't get a dataset!")
        
        trainingData, validationData, trainingLabel, validationLabel = train_test_split(dataset, labels,
                                                                            test_size = 0.20,
                                                                            random_state = 100,
                                                                            stratify = labels)
        """trainDataset = TensorDataset(torch.Tensor(trainingData), torch.LongTensor(trainingLabel))
        validationDataset = TensorDataset(torch.Tensor(validationData), torch.LongTensor(validationLabel))
        testDataset = TensorDataset(torch.Tensor(testingData), torch.LongTensor(testingLabel))      #TO DELETE?"""
        numChannels = int(confReader.getFromConf('settings','nChannels'))
        trainDataset = TensorDataset(torch.Tensor(trainingData[:, :numChannels]), torch.LongTensor(trainingLabel))
        validationDataset = TensorDataset(torch.Tensor(validationData[:, :numChannels]), torch.LongTensor(validationLabel))
        testDataset = TensorDataset(torch.Tensor(testingData[:, :numChannels]), torch.LongTensor(testingLabel))
        print("Done.")

        global exec_tag
        exec_tag = confReader.getExecutionTag()

        global minimizedValue  # save the best validation loss found
        minimizedValue = np.inf
        trainResults = ['trial', 'nEpoches', 'valLoss', 'trainLoss', 'learningRate', 'dropout', 'batchSize', 'kernel', 'num_conv', 'timeInMinutes', 'precision_validation', 'recall_validation', 'TN_validation', 'FP_validation', 'FN_validation', 'TP_validation', 'F1_validation', 'precision_test', 'recall_test', 'TN_test', 'FP_test', 'FN_test', 'TP_test', 'F1_test']
        if(int(params['loss_weight'])==1):
            trainResults.append('loss_weight')

        os.makedirs(dstPath, exist_ok=True)
        with open(f"{dstPath}trainResults{exec_tag}.csv", 'w', newline = '') as file:
            writer = csv.writer(file)
            writer.writerow(trainResults)

        #optimizing hyperparameters
        study = optuna.create_study(direction = "minimize")
        trials = int(params["trials"])
        study.optimize(lambda trial: Execution.objective(trial=trial, trainingSet=trainDataset, validationSet=validationDataset, params=params, dstPath=dstPath, testingSet=testDataset, optimizeBy=optimizeBy, size=size), n_trials = trials)
        print("Training completed.")

        best_model_state_dict = torch.load(os.path.join(dstPath,'_best_state_dict.pt'))
        model = CNN(study.best_trial.params, params, size)
        model.load_state_dict(best_model_state_dict)
        torch.save(model, f"{dstPath}CNN_{exec_tag}.pt")
        print("Best CNN model found and saved.")

        #trainResults = sorted(trainResults,
        #                     key = lambda x: x['valLoss'])
        #df = pd.DataFrame(trainResults)
        #df.to_csv(f"{dstPath}trainResults_{params['size']}.csv", index = False)
        #print('Training results saved.')
        
    """
    Funzione obiettivo che si deve ottimizzare con Optuna.
    Metodo di addestramento e valutazione di un modello CNN utilizzando una certa combinazione di iperparametri
    che serve per trovare la combinazione migliore che permette di minimizzare il validation loss.
    Ogni volta che viene trovato un modello migliore, in base al validation loss, questo viene salvato.
    Inoltre, vengono salvati anche in una lista i parametri utilizzati e i risultati ottenuti nel tentativo corrente.

    :param trial: oggetto trial che viene creato automaticamente dalla libreria Optuna che rappresenta il tentativo corrente.
    :param trainingSet: dataset contenente le immagini e le rispettive etichette su cui verrà effettuato l'addestramento.
    :param validationSet: dataset contenente le immagini e le rispettive etichette su cui verrà effettuata la validazione.
    :param params: parametri da usare durante la creazione e l'addestramento del modello CNN.
    :param dstPath: path di destinazione in cui salvare il modello CNN creato.
    """
    def objective(trial, trainingSet: TensorDataset, validationSet: TensorDataset, params: dict, dstPath: str, testingSet: TensorDataset, optimizeBy: str, size: int) -> float:
        startTime = time.time()
        global minimizedValue
        global sampled
        hyperParameters = {
            "dropout" : trial.suggest_float("dropout", 0.0, 1.0),
            "learningRate" : trial.suggest_float("learningRate", 1e-4, 1e-3),
            "batchSize" : trial.suggest_categorical("batchSize", [32, 64, 128, 256]),
            "kernel" : trial.suggest_categorical("kernel", [2, 3, 4]),
            "num_conv": trial.suggest_categorical("num_conv", [1, 2, 3]),
            # "alpha" : trial.suggest_float('alpha', 0.25, 2),
            # "gamma" : trial.suggest_float('gamma', 2, 5),
            # "reduction" : trial.suggest_categorical("reduction", ['sum', 'mean']),
        }
        if (int(params['loss_weight'])==1):
            hyperParameters['loss_weight'] = trial.suggest_float("loss_weight", .5, 1)
        trainLoader = DataLoader(trainingSet,
                                 batch_size = hyperParameters["batchSize"],
                                 shuffle = True)
        validationLoader = DataLoader(validationSet,
                                batch_size = hyperParameters["batchSize"],
                                shuffle = False)
        testLoader = DataLoader(testingSet,         #TO DELETE?
                                batch_size = hyperParameters['batchSize'],
                                shuffle = False)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f'Choosen {device} device')
        print(f'[{device}] Choosen hyperparams: {hyperParameters}')
        if optimizeBy == 'f1':
            print('Maximizing F1 Score...')
        else:
            print('Minimizing Loss...')
        model = CNN(hyperParameters, params, size).to(device)
        if (int(params['loss_weight']) == 1):
            class_0_weight = 1 - hyperParameters["loss_weight"]
            class_1_weight = hyperParameters["loss_weight"]
            class_weight = torch.FloatTensor([class_0_weight, class_1_weight]).to(device)
            lossFunct = nn.CrossEntropyLoss(class_weight)
        else:
            lossFunct = nn.CrossEntropyLoss()
        # lossFunct = sigmoid_focal_loss
        optimizer = torch.optim.Adam(model.parameters(), lr = hyperParameters["learningRate"])
        numEpochs = int(params["epoches"])
        stopTrain = EarlyStopper(patience = int(params['patience']),
                                 minDelta = float(params['minDelta']))
        best_model_state = None
        best_val_to_minimize = float('inf')

        # print(summary(model, torch.zeros((hyperParameters['batchSize'], 12, size, size))))

        for epoch in range(numEpochs):
            model.train()
            trainLoss = 0.0
            for images, classes in tqdm(trainLoader, desc = f"Trial {trial.number} - Training model #{epoch}", leave = False):
                images = images.to(device)
                classes = classes.to(device)
                #forward propagation
                optimizer.zero_grad()
                outputs = model(images)
                loss = lossFunct(outputs, classes)
                # aggiunta per sigmoid
                # classes= classes.float()
                # loss = lossFunct(outputs, classes.view(-1, 1), alpha=hyperParameters['alpha'], gamma=hyperParameters['gamma'], reduction=hyperParameters['reduction'])
                #backward propagation and optimize
                loss.backward()
                optimizer.step()
                trainLoss += loss.item()
            with torch.no_grad():
                model.eval()
                valLoss = 0.0
                valPredictions = []
                valClasses = []
                for images, classes in tqdm(validationLoader, desc = f"Trial {trial.number} - Validation loss", leave = False):
                    images = images.to(device)
                    classes = classes.to(device)
                    outputs = model(images)
                    loss = lossFunct(outputs, classes)
                    # aggiunta per sigmoid
                    #classes = classes.float()
                    #loss = lossFunct(outputs, classes.view(-1, 1), alpha=hyperParameters['alpha'], gamma=hyperParameters['gamma'], reduction=hyperParameters['reduction'])
                    valLoss += loss.item()
                    valPredictions.append(outputs.cpu().numpy())
                    valClasses.append(classes.cpu().numpy())
                val_pred = np.argmax(np.concatenate(valPredictions), axis = 1)
                # val_pred = (np.concatenate(valPredictions) >= 0.5).astype(int)
                targets_app = np.concatenate(valClasses)
            val_cm = confusion_matrix(targets_app, val_pred)
            tn, fp, fn, tp = val_cm.ravel()
            pNormal = 0 if (tp + fp) == 0 else tp / (tp + fp)
            rNormal = 0 if (tp + fn) == 0 else tp / (tp + fn)
            f1ValNormal = 0 if pNormal + rNormal == 0 else 2 * ((pNormal * rNormal) / (pNormal + rNormal))
            #calculate train and validation loss of the epoch
            trainingLoss = trainLoss / len(trainLoader)
            validationLoss = valLoss / len(validationLoader)
            if optimizeBy == 'f1':
                toMinimizeVal = -f1ValNormal
            else:
                toMinimizeVal = validationLoss

            if toMinimizeVal < best_val_to_minimize:
                print(f'Found new best {-best_val_to_minimize} => {-toMinimizeVal}')
                best_val_to_minimize = toMinimizeVal
                best_model_state = model.state_dict()
                torch.save(best_model_state,os.path.join(dstPath,'_state_dict.pt'))

            print(f"\nTrial/Epoch: {trial.number}/{epoch} - Validation F1: {f1ValNormal:.5f} - TrainLoss: {trainingLoss:.5f} - ValLoss: {validationLoss:.5f}")
            if stopTrain.earlyStop(toMinimizeVal):
                break
        endTime = time.time()

        best_model_state = torch.load(os.path.join(dstPath,'_state_dict.pt'))
        if best_model_state is not None:
            model.load_state_dict(best_model_state)


        with torch.no_grad():
            # validation
            model.eval()
            allPredictions = []
            allClasses = []
            for images, classes in tqdm(validationLoader, desc = f"Trial {trial.number} - Validation model", leave = False):
                images = images.to(device)
                classes = classes.to(device)
                outputs = model(images)
                allPredictions.append(outputs.cpu().numpy())
                allClasses.append(classes.cpu().numpy())
            allPredictions = np.concatenate(allPredictions)
            allClasses = np.concatenate(allClasses)
            prediction = np.argmax(allPredictions, axis = 1)
            # prediction = (np.concatenate(allPredictions) >= 0.5).astype(int)
            cm = confusion_matrix(allClasses, prediction)
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * ((precision * recall) / (precision + recall))
            # test
            allPredictions = []
            allClasses = []
            for images, classes in tqdm(testLoader, desc = f"Trial {trial.number} - Testing model", leave = False):
                images = images.to(device)
                classes = classes.to(device)
                outputs = model(images)
                allPredictions.append(outputs.cpu().numpy())
                allClasses.append(classes.cpu().numpy())
            allPredictions = np.concatenate(allPredictions)
            allClasses = np.concatenate(allClasses)
            prediction = np.argmax(allPredictions, axis = 1)
            # prediction = (np.concatenate(allPredictions) >= 0.5).astype(int)
            cm_test = confusion_matrix(allClasses, prediction)
            tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
            precision_test = tp_test / (tp_test + fp_test)
            recall_test = tp_test / (tp_test + fn_test)
            f1_test = 2 * ((precision_test * recall_test) / (precision_test + recall_test))
            
        trainResults = {
            'trial' : trial.number,
            'nEpoches': epoch + 1,
            'valLoss' : validationLoss,
            'trainLoss' : trainingLoss,
            'learningRate' : hyperParameters['learningRate'],
            'dropout' : hyperParameters['dropout'],
            'batchSize' : hyperParameters['batchSize'],
            'kernel' : hyperParameters['kernel'],
            'num_conv' : hyperParameters['num_conv'],
            # 'alpha': hyperParameters['alpha'],
            # 'gamma': hyperParameters['gamma'],
            # 'reduction': hyperParameters['reduction'],
            'time' : (endTime - startTime) / 60 ,
            'precision_validation' : precision,
            'recall_validation' : recall,
            'TN_validation' : tn,
            'FP_validation' : fp,
            'FN_validation' : fn,
            'TP_validation' : tp,
            'F1_validation' : f1,
            'precision_test' : precision_test,
            'recall_test' : recall_test,
            'TN_test' : tn_test,
            'FP_test' : fp_test,
            'FN_test' : fn_test,
            'TP_test' : tp_test,
            'F1_test' : f1_test,
        }

        if (int(params['loss_weight']) == 1):
            trainResults['loss_weight'] = class_1_weight


        #updating train results
        with open(f"{dstPath}trainResults{exec_tag}.csv", 'a', newline = '') as file:
            writer = csv.DictWriter(file, fieldnames = trainResults.keys())
            writer.writerow(trainResults)

        #saving the best model found
        if best_val_to_minimize < minimizedValue:
            minimizedValue = best_val_to_minimize
            try:
                torch.save(best_model_state,os.path.join(dstPath,'_best_state_dict.pt'))
                # torch.save(model, f"{dstPath}CNN{sampled}_{size}_{optimizeBy}.pt")
                print("New best CNN model found and saved.")
            except:
                raise IOError("Error during saving the model CNN")

        print(f'====== To minimize val finale: {best_val_to_minimize} ======')
        return best_val_to_minimize
    

    """
    Metodo che si occupa del testing di un modello CNN precedentemente creato e addestrato.
    Acquisisce il testSet a partire dal percorso sorgente ed esegue il testing del modello su
    questi dati, calcolando diverse metriche e le salva in un file csv nel percorso di destinazione
    specificato in output.

    :param testPath: percorso sorgente contenente il test set su cui testare il modello.
    :param modelPath: percorso contenente un modello precedentemente addestrato e salvato; percorso in cui salvare
                      anche i risultati del test.
    :param batchSize: dimensione del batch da utilizzare durante il testing.
    :param size: dimensione delle immagini neighbour su cui è stato addestrato il modello da testare.
    """
    def testCNN(testPath: str, modelPath: str, batchSize: int, size: int, numChannels: int, confReader:any) -> None:
        print("Getting a testSet... ", end = ' ')
        try:
            with open(f"{testPath}testingData_{size}.pickle", "rb") as file:
                testData = pickle.load(file)
            with open(f"{testPath}testingLabels_{size}.pickle", "rb") as file:
                testLabels = pickle.load(file)
        except:
            raise IOError("Can't get the testSet.")
        # numChannels = int(confReader.getFromConf('settings', 'nChannels'))
        testSet = TensorDataset(torch.Tensor(testData[:, :numChannels]), torch.LongTensor(testLabels))
        testLoader = DataLoader(testSet,
                                batch_size = batchSize,
                                shuffle = False)
        print('Done.')
        print('Open a model CNN... ', end = ' ')
        exec_tag = confReader.getExecutionTag()
        try:
            cnn = torch.load(os.path.join(modelPath, f"CNN_{exec_tag}.pt"))
        except:
            raise IOError("Can't get a model CNN")
        print('Done')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cnn = cnn.to(device)
        with torch.no_grad():
            cnn.eval()
            allPredictions = []
            allClasses = []
            for images, classes in tqdm(testLoader, desc = f"Testing model", leave = False):
                images = images.to(device)
                classes = classes.to(device)
                outputs = cnn(images)
                allPredictions.append(outputs.cpu().numpy())
                allClasses.append(classes.cpu().numpy())
            allPredictions = np.concatenate(allPredictions)
            allClasses = np.concatenate(allClasses)
            prediction = np.argmax(allPredictions, axis = 1)
            cm = confusion_matrix(allClasses, prediction)
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * ((precision * recall) / (precision + recall))
            oa = (tn + tp)/(tn + fp + fn + tp)
            pNormal = tn / (tn + fn)
            rNormal = tn / (tn + fp)
            f1Normal = 2 * ((pNormal * rNormal) / (pNormal + rNormal))
            macroF1 = (f1 + f1Normal) / 2
            weighted = (f1 * ((tp + fn) / (tn + fp + fn + tp))) + (f1Normal * ((tn + fp) / (tn + fp + fn + tp)))
            gMean = np.sqrt(recall * rNormal)
            iou = tp / (tp + fp + fn)
            aa = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
            fdr = fp / (fp + tp)
            mar = fn / (fn + tp)
        testResults = {
            'Precision' : precision,
            'Recall' : recall,
            'TN' : tn,
            'FP' : fp,
            'FN' : fn,
            'TP' : tp,
            'F1' : f1,
            'OA' : oa,
            'P_normal' : pNormal,
            'R_normal' : rNormal,
            'F1_normal' : f1Normal,
            'Macro_F1' : macroF1,
            'Weighted' : weighted,
            'G_Mean' : gMean,
            'IOU' : iou,
            'AA' : aa,
            'FDR' : fdr,
            'MAR' : mar
        }
        #writing the csv file
        with open(f"{modelPath}testResults_{exec_tag}.csv", 'w', newline = '') as file:
            fieldNames = testResults.keys()
            writer = csv.DictWriter(file, fieldnames = fieldNames)
            writer.writeheader()
            writer.writerow(testResults)
        print('Testing completed.\nTest results saved.')


    def testSingleImagesCNN(testPath: str, modelPath: str, batchSize: int, size: int, numChannels: int, confReader:any) -> None:
        print('Getting CNN model... ', end=' ')
        exec_tag = confReader.getExecutionTag()
        try:
            cnn = torch.load(os.path.join(modelPath, f"CNN_{exec_tag}.pt"))
        except:
            raise IOError("Can't get a model CNN")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cnn = cnn.to(device)
        print('Loaded CNN model...')

        single_data_path = os.path.join(testPath, 'single_test', 'data')
        single_labels_path = os.path.join(testPath, 'single_test', 'mask')

        filelist=[]
        list_test_results = []
        for filename in os.listdir(single_data_path):
            if filename.endswith(".pickle") and "_" in filename and "x" in filename:
                filelist.append(filename)

        for data_file_name in filelist:
            img_id = data_file_name.split('_')[0]
            width = int(data_file_name.split('_')[1].split('.')[0].split('x')[0])
            height = int(data_file_name.split('_')[1].split('.')[0].split('x')[1])
            try:
                with open(os.path.join(single_data_path, data_file_name), "rb") as file:
                    testData = pickle.load(file)
                with open(os.path.join(single_labels_path, data_file_name), "rb") as file:
                    testLabels = pickle.load(file)
            except:
                raise IOError(f"Can't get {data_file_name}")

            testSet = TensorDataset(torch.Tensor(testData[:, :numChannels]), torch.LongTensor(testLabels))
            testLoader = DataLoader(testSet,
                                    batch_size = batchSize,
                                    shuffle = False)

            with torch.no_grad():
                cnn.eval()
                allPredictions = []
                allAttPixel = []
                allAttCh = []
                allClasses = []
                for images, classes in tqdm(testLoader, desc = f"Testing model", leave = False):
                    images = images.to(device)
                    classes = classes.to(device)
                    outputs, px_att_map, ch_att_map = cnn(images, get_explanation=True)
                    allPredictions.append(outputs.cpu().numpy())
                    allAttPixel.append(px_att_map.cpu().numpy())
                    allClasses.append(classes.cpu().numpy())
                allPredictions = np.concatenate(allPredictions)
                allAttPixel = np.concatenate(allAttPixel)
                allClasses = np.concatenate(allClasses)
                prediction = np.argmax(allPredictions, axis = 1)
                cm = confusion_matrix(allClasses, prediction)
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * ((precision * recall) / (precision + recall))
                oa = (tn + tp)/(tn + fp + fn + tp)
                pNormal = tn / (tn + fn)
                rNormal = tn / (tn + fp)
                f1Normal = 2 * ((pNormal * rNormal) / (pNormal + rNormal))
                macroF1 = (f1 + f1Normal) / 2
                weighted = (f1 * ((tp + fn) / (tn + fp + fn + tp))) + (f1Normal * ((tn + fp) / (tn + fp + fn + tp)))
                gMean = np.sqrt(recall * rNormal)
                iou = tp / (tp + fp + fn)
                aa = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
                fdr = fp / (fp + tp)
                mar = fn / (fn + tp)
            testResults = {
                'id': img_id,
                'Precision' : precision,
                'Recall' : recall,
                'TN' : tn,
                'FP' : fp,
                'FN' : fn,
                'TP' : tp,
                'F1' : f1,
                'OA' : oa,
                'P_normal' : pNormal,
                'R_normal' : rNormal,
                'F1_normal' : f1Normal,
                'Macro_F1' : macroF1,
                'Weighted' : weighted,
                'G_Mean' : gMean,
                'IOU' : iou,
                'AA' : aa,
                'FDR' : fdr,
                'MAR' : mar
            }
            list_test_results.append(testResults)

            savePredData = int(confReader.getFromConf('options', 'savePredData')) == 1

            if savePredData:
                data_dest_dir = os.path.join(modelPath, f'singlePredImages_{exec_tag}', 'data', img_id)
                os.makedirs(data_dest_dir, exist_ok=True)
                with open(os.path.join(data_dest_dir,'pixel-attention.pickle'), 'wb') as file_pickle:
                    pickle.dump(allAttPixel, file_pickle)
                with open(os.path.join(data_dest_dir,'predictions.pickle'), 'wb') as file_pickle:
                    pickle.dump(prediction, file_pickle)
                with open(os.path.join(data_dest_dir,'ground-truth.pickle'), 'wb') as file_pickle:
                    pickle.dump(allClasses, file_pickle)

                sizes = {'width': width, 'height': height}
                with open(os.path.join(data_dest_dir, 'image_size.json'), "w") as file:
                    json.dump(sizes, file)

                img_dest_dir = os.path.join(modelPath, f'singlePredImages_{exec_tag}')
                os.makedirs(img_dest_dir, exist_ok=True)

                og_matrix = np.reshape(allClasses, (height, width))
                og_matrix[og_matrix == 1] = 255
                im = ImagePIL.fromarray(og_matrix.astype(np.uint8))
                im.save(os.path.join(img_dest_dir, f'original_{img_id}.png'))

                pred_matrix = np.reshape(prediction, (height, width))
                pred_matrix[pred_matrix == 1] = 255
                im = ImagePIL.fromarray(pred_matrix.astype(np.uint8))
                im.save(os.path.join(img_dest_dir, f'predicted_{img_id}.png'))


        df_single_res = pd.DataFrame(list_test_results)
        df_single_res.to_csv(os.path.join(modelPath,f'singleTestResults_{exec_tag}.csv'), index=False, sep='\t')
        print('Testing completed.\nTest results saved.')


    def makeExplanation(confReader: any, modelPath: str) -> None:
        exec_tag = confReader.getExecutionTag()
        data_dir = os.path.join(modelPath, f'singlePredImages_{exec_tag}', 'data')

        list_max = []
        list_min = []
        for img_id in os.listdir(data_dir):
            data_img_dir = os.path.join(data_dir, str(img_id))
            attention_path = os.path.join(data_img_dir, 'pixel-attention.pickle')
            if os.path.exists(attention_path):
                with open(attention_path, 'rb') as file_pickle:
                    attention_maps = pickle.load(file_pickle)
                    list_min.append(attention_maps.min())
                    list_max.append(attention_maps.max())
        min_val = min(list_min)
        max_val = max(list_max)
        print(f'Global min: {min_val} - Global max: {max_val}')

        local_min_max_list = []

        # lista di tutte le attention map del test set
        whole_test_attention_maps = []
        whole_test_prediction = []
        whole_test_groundtruth = []

        whole_pure_0_vect = []
        whole_spure_0_vect = []
        whole_pure_1_vect = []
        whole_spure_1_vect = []
        whole_pure_0_vect_pred = []
        whole_spure_0_vect_pred = []
        whole_pure_1_vect_pred = []
        whole_spure_1_vect_pred = []

        base_explanation_dir = os.path.join(modelPath, f'singlePredImages_{exec_tag}', 'explanation_dir')
        for img_id in os.listdir(data_dir):
            data_img_dir = os.path.join(data_dir, str(img_id))
            attention_path = os.path.join(data_img_dir, 'pixel-attention.pickle')
            if os.path.exists(attention_path):
                explanation_dir = os.path.join(base_explanation_dir, str(img_id))
                os.makedirs(explanation_dir, exist_ok=True)
                data_img_dir = os.path.join(data_dir, str(img_id))
                with open(os.path.join(data_img_dir, 'pixel-attention.pickle'), 'rb') as file_pickle:
                    attention_maps = pickle.load(file_pickle)
                with open(os.path.join(data_img_dir, 'ground-truth.pickle'), 'rb') as file_pickle:
                    predictions = pickle.load(file_pickle)
                with open(os.path.join(data_img_dir, 'ground-truth.pickle'), 'rb') as file_pickle:
                    original_classes = pickle.load(file_pickle)
                with open(os.path.join(data_img_dir, "image_size.json"), "r") as file:
                    image_size = json.load(file)

                img_min_val = attention_maps.min()
                img_max_val = attention_maps.max()

                local_min_max_list.append({
                    'img_id': img_id,
                    'min': img_min_val,
                    'max': img_max_val
                })

                pred_tiled = extract_neigh_from_predictions(original_matrix=np.reshape(predictions, (image_size.get('height'), image_size.get('width'))), window_size=5)
                gt_tiled = extract_neigh_from_predictions(original_matrix=np.reshape(original_classes, (image_size.get('height'), image_size.get('width'))), window_size=5)

                global_scaled_attentions = (attention_maps - min_val) / (max_val - min_val)
                scaled_attentions = (attention_maps - img_min_val) / (img_max_val - img_min_val)

                att_full_scene = newComputeAttentionScene(attention_tensor=np.reshape(scaled_attentions, (image_size.get('height'), image_size.get('width'), 5, 5)),
                                                       ground_truth_matrix=np.reshape(original_classes, (image_size.get('height'), image_size.get('width'))),
                                                       k=2)
                save_from_scaled_attention_map(att_full_scene, os.path.join(explanation_dir, 'mean_full_attention_map.png'))

                att_full_scene = newComputeAttentionScene(attention_tensor=np.reshape(scaled_attentions, (image_size.get('height'), image_size.get('width'), 5, 5)),
                                                       ground_truth_matrix=np.reshape(predictions, (image_size.get('height'), image_size.get('width'))),
                                                       k=2)
                save_from_scaled_attention_map(att_full_scene, os.path.join(explanation_dir, 'mean_full_attention_map_pred.png'))

                pure_0_matrix, spure_0_matrix, pure_1_matrix, spure_1_matrix = computePixelBoundary(mask_matrix=np.reshape(original_classes, (image_size.get('height'), image_size.get('width'))), k=2)
                pure_0_vect = pure_0_matrix.reshape(-1)
                spure_0_vect = spure_0_matrix.reshape(-1)
                pure_1_vect = pure_1_matrix.reshape(-1)
                spure_1_vect = spure_1_matrix.reshape(-1)

                pred_pure_0_matrix, pred_spure_0_matrix, pred_pure_1_matrix, pred_spure_1_matrix = computePixelBoundary(mask_matrix=np.reshape(predictions, (image_size.get('height'), image_size.get('width'))), k=2)
                pred_pure_0_vect = pred_pure_0_matrix.reshape(-1)
                pred_spure_0_vect = pred_spure_0_matrix.reshape(-1)
                pred_pure_1_vect = pred_pure_1_matrix.reshape(-1)
                pred_spure_1_vect = pred_spure_1_matrix.reshape(-1)

                # calcolo attention map locali (size x size)
                mean_attention_map = np.mean(scaled_attentions, axis=0)
                mean_attention_map_0_gt = np.mean(scaled_attentions[original_classes==0], axis=0)
                mean_attention_map_1_gt = np.mean(scaled_attentions[original_classes==1], axis=0)
                mean_attention_map_0_pred = np.mean(scaled_attentions[predictions == 0], axis=0)
                mean_attention_map_1_pred = np.mean(scaled_attentions[predictions == 1], axis=0)
                mean_pure_0_attentions = np.mean(scaled_attentions[pure_0_vect==1], axis=0)
                mean_spure_0_attentions = np.mean(scaled_attentions[spure_0_vect==1], axis=0)
                mean_pure_1_attentions = np.mean(scaled_attentions[pure_1_vect==1], axis=0)
                mean_spure_1_attentions = np.mean(scaled_attentions[spure_1_vect==1], axis=0)
                mean_pure_0_attentions_pred = np.mean(scaled_attentions[pred_pure_0_vect == 1], axis=0)
                mean_spure_0_attentions_pred = np.mean(scaled_attentions[pred_spure_0_vect == 1], axis=0)
                mean_pure_1_attentions_pred = np.mean(scaled_attentions[pred_pure_1_vect == 1], axis=0)
                mean_spure_1_attentions_pred = np.mean(scaled_attentions[pred_spure_1_vect == 1], axis=0)
                random_pure_0_attentions = np.zeros((5, 5)) if len(scaled_attentions[pure_0_vect == 1]) == 0 else random.choice(scaled_attentions[pure_0_vect == 1])
                random_spure_0_attentions = np.zeros((5, 5)) if len(scaled_attentions[spure_0_vect == 1]) == 0 else random.choice(scaled_attentions[spure_0_vect == 1])
                random_pure_1_attentions = np.zeros((5, 5)) if len(scaled_attentions[pure_1_vect == 1]) == 0 else random.choice(scaled_attentions[pure_1_vect == 1])
                random_spure_1_attentions = np.zeros((5, 5)) if len(scaled_attentions[spure_1_vect == 1]) == 0 else random.choice(scaled_attentions[spure_1_vect == 1])

                pred_pure_0_att= scaled_attentions[(pure_0_vect == 1) & (original_classes == 0) & (predictions == 0)]
                pred_pure_1_att= scaled_attentions[(pure_1_vect == 1) & (original_classes == 1) & (predictions == 1)]
                pred_spure_0_att= scaled_attentions[(spure_0_vect == 1) & (original_classes == 0) & (predictions == 0)]
                pred_spure_1_att= scaled_attentions[(spure_1_vect == 1) & (original_classes == 1) & (predictions == 1)]

                pred_pure_0_tiled = gt_tiled[(pure_0_vect == 1) & (original_classes == 0) & (predictions == 0)]
                pred_pure_1_tiled = gt_tiled[(pure_1_vect == 1) & (original_classes == 1)  & (predictions == 1)]
                pred_spure_0_tiled = gt_tiled[(spure_0_vect == 1) & (original_classes == 0) & (predictions == 0)]
                pred_spure_1_tiled = gt_tiled[(spure_1_vect == 1) & (original_classes == 1) & (predictions == 1)]


                pred_random_pure_0_attentions = np.zeros((5, 5)) if len(pred_pure_0_att) == 0 else random.choice(pred_pure_0_att)
                pred_random_spure_0_attentions = np.zeros((5, 5)) if len(pred_spure_0_att) == 0 else random.choice(pred_spure_0_att)
                pred_random_pure_1_attentions = np.zeros((5, 5)) if len(pred_pure_1_att) == 0 else random.choice(pred_pure_1_att)
                pred_random_spure_1_attentions = np.zeros((5, 5)) if len(pred_spure_1_att) == 0 else random.choice(pred_spure_1_att)

                # salvo le immagini delle attention map locali (sizexsize)
                save_from_scaled_attention_map(mean_attention_map, os.path.join(explanation_dir, 'mean_attention_map.png'), f'Img {img_id}')
                save_from_scaled_attention_map(mean_attention_map_0_gt, os.path.join(explanation_dir, 'mean_attention_map_0_gt.png'))
                save_from_scaled_attention_map(mean_attention_map_1_gt, os.path.join(explanation_dir, 'mean_attention_map_1_gt.png'))
                save_from_scaled_attention_map(mean_attention_map_0_pred, os.path.join(explanation_dir, 'mean_attention_map_0_pred.png'))
                save_from_scaled_attention_map(mean_attention_map_1_pred, os.path.join(explanation_dir, 'mean_attention_map_1_pred.png'))
                # gt
                save_from_scaled_attention_map(mean_pure_0_attentions, os.path.join(explanation_dir, 'mean_pure_0_attentions.png'))
                save_from_scaled_attention_map(mean_spure_0_attentions, os.path.join(explanation_dir, 'mean_spure_0_attentions.png'))
                save_from_scaled_attention_map(mean_pure_1_attentions, os.path.join(explanation_dir, 'mean_pure_1_attentions.png'))
                save_from_scaled_attention_map(mean_spure_1_attentions, os.path.join(explanation_dir, 'mean_spure_1_attentions.png'))
                save_from_scaled_attention_map(random_pure_0_attentions, os.path.join(explanation_dir, 'random_pure_0_attentions.png'))
                save_from_scaled_attention_map(random_spure_0_attentions, os.path.join(explanation_dir, 'random_spure_0_attentions.png'))
                save_from_scaled_attention_map(random_pure_1_attentions, os.path.join(explanation_dir, 'random_pure_1_attentions.png'))
                save_from_scaled_attention_map(random_spure_1_attentions, os.path.join(explanation_dir, 'random_spure_1_attentions.png'))
                #pred
                save_from_scaled_attention_map(mean_pure_0_attentions_pred, os.path.join(explanation_dir, 'mean_pure_0_attentions-pred.png'))
                save_from_scaled_attention_map(mean_spure_0_attentions_pred, os.path.join(explanation_dir, 'mean_spure_0_attentions-pred.png'))
                save_from_scaled_attention_map(mean_pure_1_attentions_pred, os.path.join(explanation_dir, 'mean_pure_1_attentions-pred.png'))
                save_from_scaled_attention_map(mean_spure_1_attentions_pred, os.path.join(explanation_dir, 'mean_spure_1_attentions-pred.png'))
                save_from_scaled_attention_map(pred_random_pure_0_attentions, os.path.join(explanation_dir, 'random_pure_0_attentions-pred.png'))
                save_from_scaled_attention_map(pred_random_spure_0_attentions, os.path.join(explanation_dir, 'random_spure_0_attentions-pred.png'))
                save_from_scaled_attention_map(pred_random_pure_1_attentions, os.path.join(explanation_dir, 'random_pure_1_attentions-pred.png'))
                save_from_scaled_attention_map(pred_random_spure_1_attentions, os.path.join(explanation_dir, 'random_spure_1_attentions-pred.png'))

                os.makedirs(os.path.join(explanation_dir,'all_pure_pred_1_gt_1'), exist_ok=True)
                idx = 0
                for mat, pred in zip(pred_pure_1_att, pred_pure_1_tiled):
                    save_from_scaled_attention_map_with_label(normalized_arr=mat, save_path=os.path.join(explanation_dir, 'all_pure_pred_1_gt_1', f'{idx}.png'), label_attention=pred)
                    idx+=1

                os.makedirs(os.path.join(explanation_dir, 'all_spure_pred_1_gt_1'), exist_ok=True)
                idx = 0
                for mat, pred in zip(pred_spure_1_att, pred_spure_1_tiled):
                    save_from_scaled_attention_map_with_label(normalized_arr=mat, save_path=os.path.join(explanation_dir, 'all_spure_pred_1_gt_1', f'{idx}.png'), label_attention=pred)
                    idx+=1

                os.makedirs(os.path.join(explanation_dir, 'all_spure_pred_0_gt_0'), exist_ok=True)
                idx = 0
                for mat, pred in zip(pred_spure_0_att, pred_spure_0_tiled):
                    save_from_scaled_attention_map_with_label(normalized_arr=mat, save_path=os.path.join(explanation_dir, 'all_spure_pred_0_gt_0', f'{idx}.png'), label_attention=pred)
                    idx += 1

                whole_test_attention_maps.extend(global_scaled_attentions)
                whole_test_prediction.extend(predictions)
                whole_test_groundtruth.extend(original_classes)
                whole_pure_0_vect.extend(pure_0_vect)
                whole_spure_0_vect.extend(spure_0_vect)
                whole_pure_1_vect.extend(pure_1_vect)
                whole_spure_1_vect.extend(spure_1_vect)
                whole_pure_0_vect_pred.extend(pred_pure_0_vect)
                whole_spure_0_vect_pred.extend(pred_spure_0_vect)
                whole_pure_1_vect_pred.extend(pred_pure_1_vect)
                whole_spure_1_vect_pred.extend(pred_spure_1_vect)

        whole_test_attention_maps = np.array(whole_test_attention_maps)
        whole_test_prediction = np.array(whole_test_prediction)
        whole_test_groundtruth = np.array(whole_test_groundtruth)
        whole_pure_0_vect = np.array(whole_pure_0_vect)
        whole_spure_0_vect = np.array(whole_spure_0_vect)
        whole_pure_1_vect = np.array(whole_pure_1_vect)
        whole_spure_1_vect = np.array(whole_spure_1_vect_pred)
        whole_pure_0_vect_pred = np.array(whole_pure_0_vect_pred)
        whole_spure_0_vect_pred = np.array(whole_spure_0_vect_pred)
        whole_pure_1_vect_pred = np.array(whole_pure_1_vect_pred)
        whole_spure_1_vect_pred = np.array(whole_spure_1_vect_pred)

        # calcolo attention map globali (size x size)
        mean_attention_map = np.mean(whole_test_attention_maps, axis=0)
        mean_attention_map_0_gt = np.mean(whole_test_attention_maps[whole_test_groundtruth == 0], axis=0)
        mean_attention_map_1_gt = np.mean(whole_test_attention_maps[whole_test_groundtruth == 1], axis=0)
        mean_attention_map_0_pred = np.mean(whole_test_attention_maps[whole_test_prediction == 0], axis=0)
        mean_attention_map_1_pred = np.mean(whole_test_attention_maps[whole_test_prediction == 1], axis=0)
        mean_pure_0_attentions = np.mean(whole_test_attention_maps[whole_pure_0_vect == 1], axis=0)
        mean_spure_0_attentions = np.mean(whole_test_attention_maps[whole_spure_0_vect == 1], axis=0)
        mean_pure_1_attentions = np.mean(whole_test_attention_maps[whole_pure_1_vect == 1], axis=0)
        mean_spure_1_attentions = np.mean(whole_test_attention_maps[whole_spure_1_vect == 1], axis=0)
        mean_pure_0_attentions_pred = np.mean(whole_test_attention_maps[whole_pure_0_vect_pred == 1], axis=0)
        mean_spure_0_attentions_pred = np.mean(whole_test_attention_maps[whole_spure_0_vect_pred == 1], axis=0)
        mean_pure_1_attentions_pred = np.mean(whole_test_attention_maps[whole_pure_1_vect_pred == 1], axis=0)
        mean_spure_1_attentions_pred = np.mean(whole_test_attention_maps[whole_spure_1_vect_pred == 1], axis=0)
        # salvo le immagini delle attention map locali (sizexsize)
        save_from_scaled_attention_map(mean_attention_map, os.path.join(base_explanation_dir, 'mean_attention_map.png'))
        save_from_scaled_attention_map(mean_attention_map_0_gt, os.path.join(base_explanation_dir, 'mean_attention_map_0_gt.png'))
        save_from_scaled_attention_map(mean_attention_map_1_gt, os.path.join(base_explanation_dir, 'mean_attention_map_1_gt.png'))
        save_from_scaled_attention_map(mean_attention_map_0_pred, os.path.join(base_explanation_dir, 'mean_attention_map_0_pred.png'))
        save_from_scaled_attention_map(mean_attention_map_1_pred, os.path.join(base_explanation_dir, 'mean_attention_map_1_pred.png'))
        save_from_scaled_attention_map(mean_pure_0_attentions, os.path.join(base_explanation_dir, 'mean_pure_0_attentions.png'))
        save_from_scaled_attention_map(mean_spure_0_attentions, os.path.join(base_explanation_dir, 'mean_spure_0_attentions.png'))
        save_from_scaled_attention_map(mean_pure_1_attentions, os.path.join(base_explanation_dir, 'mean_pure_1_attentions.png'))
        save_from_scaled_attention_map(mean_spure_1_attentions, os.path.join(base_explanation_dir, 'mean_spure_1_attentions.png'))
        save_from_scaled_attention_map(mean_pure_0_attentions_pred, os.path.join(base_explanation_dir, 'mean_pure_0_attentions-pred.png'))
        save_from_scaled_attention_map(mean_spure_0_attentions_pred, os.path.join(base_explanation_dir, 'mean_spure_0_attentions-pred.png'))
        save_from_scaled_attention_map(mean_pure_1_attentions_pred, os.path.join(base_explanation_dir, 'mean_pure_1_attentions-pred.png'))
        save_from_scaled_attention_map(mean_spure_1_attentions_pred, os.path.join(base_explanation_dir, 'mean_spure_1_attentions-pred.png'))

        dfminmax = pd.DataFrame(local_min_max_list)
        dfminmax.to_csv(os.path.join(base_explanation_dir,'local_min_max_attentions.csv'), index=False, sep='\t', header=True)
        print('Maked Attentions...')


    def normalizeImages (dsConf, size: int) -> None:
        sourceDir = dsConf['imagesPath']
        normSourceDir = dsConf['normImagesPath']
        print(f'Saving Normalized images in {normSourceDir}')
        destinationDir = dsConf['neighbourPath']
        os.makedirs(destinationDir, exist_ok = True)
        os.makedirs(normSourceDir, exist_ok = True)
        trainBound = int(dsConf['trainBound'])
        scaler = MinMaxScaler()
        for fileName in tqdm(os.listdir(sourceDir), desc = "Fitting Scaler"):
            if fileName.endswith('tif') and int(fileName.split('.')[0].split('_')[1]) <= trainBound:
                image = Image(f"{sourceDir}{fileName}")
                bands = image.getAllPixel()
                num_bands, num_rows, num_cols = bands.shape
                bands_matrix = bands.reshape(num_bands, num_rows * num_cols).T
                bands_matrix = add_vegetation_features(add_scl_feature(bands_matrix))
                scaler = scaler.partial_fit(bands_matrix)
        scaler_path = f"{dsConf['datasetPath']}data_minmax_scaler.pkl"
        joblib.dump(scaler,scaler_path)

        for fileName in tqdm(os.listdir(sourceDir), desc="Scaling images"):
            if fileName.endswith('tif'):
                image = Image(f"{sourceDir}{fileName}")
                bands = copy(image.getAllPixel()) #13,153,225
                num_bands, num_rows, num_cols = bands.shape
                bands_matrix = bands.reshape(num_bands, num_rows * num_cols).T
                bands_matrix = add_vegetation_features(add_scl_feature(bands_matrix))
                scaled_bands_matrix = scaler.transform(bands_matrix)
                scaled_bands = scaled_bands_matrix.T.reshape(num_bands+3,num_rows,num_cols)
                scaled_image_path = f"{normSourceDir}{fileName}"
                # tifffile.imwrite(scaled_image_path, scaled_bands)
                createGeoTiff(scaled_bands,scaled_image_path, image.getTransform())
        print("Neighbour Images generated.")