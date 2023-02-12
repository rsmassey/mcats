import numpy as np
import pandas as pd
import os

from logistic_model import train_and_classify
import audio_to_df
import feature_extraction
from file_to_prediction import file_to_prediction

from ml_logic.params import (
    LOCAL_DATA_PATH,
    LOCAL_SONG_PATH
)

if __name__ == '__main__':

    #X is rock which is 7
    #X = [0.35286835736042205,4.209018291530816,-4.139893281373414,-0.36151849976613537,-0.324048278541866,-0.0220056891316001,3.212987597216296,-0.06988163698286191,3.687788756573567,-0.8068050832828481,2.09099896879914,0.12455206251666981,2.8137793128025876,0.8445060683875784,1.0404135772669734,-1.0342002325426003,0.8374341972447673,1.2213124090210183,0.269077953711084,-1.2811136245806218,2.1692740843216702,0.958075004977099,0.7884862891323723,-1.6959285821925387,0.2038282950946665,-0.13479373421641466,0.5722168754367878,-0.991672033163992,0.2573896177949083,0.12379604003811286,0.3561963557079899,-1.6057797354532017,0.3981013689321286,-0.10590154820221166,-0.3247620464197768,-1.5495749389009055,0.02751838896081336,0.5544470708956186,-0.21232181042115794,-0.834468436244732,-0.25262895316696476,-0.2682830529352942,-0.4531911543703119,-1.4978249384100708,0.16634324899234657,-0.9228654629196373,-0.04444391263880561,-0.6813035323056931,-0.3021834503055795,0.16608864320890349,-0.2448159703721963,-0.32982132671195785,-0.49226571630971333,-0.3918646889383616,0.018836399049240594,0.18458706090893628,-0.2439589039123528,-0.6858964019059938,-0.45167158337697055,-1.134074747775978,-0.40968465821791306,-0.26531265226949685,-0.8709209127561559,1.1915202977521062,-0.18383049748419786,2.0226116286297247,0.2952246055329907,1.13531058199389,-0.28508273004512624,-0.22881120524536436,-0.34092160243209746,0.15607086977922408,0.010026071633334025,0.8940469090158178,0.17761881500814497,2.773594997582999,0.1985018549490116,3.484481876208549,0.35047900603311644,2.6054772906919785,0.2773489981734091,0.6058823969171808,0.3639214827512356,-0.041324527387303994,0.2548780611263805,-0.04199178413029917,-0.28951039315162425,0.9006024012779197,-0.3590705198955154]

    X = file_to_prediction(LOCAL_SONG_PATH)

    try:
        print('Let\'s go the app is thinking !!')



        #Transform  the audio input into usable data

        #Preprocess the raw data


        #train a model and use the provided input to classify
        result = train_and_classify(X)
        print(f'The genre of the provided song is: {result}')

    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)