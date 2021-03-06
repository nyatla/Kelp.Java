﻿package jp.nyatla.kelpjava;

import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.loss.LossFunction;
import jp.nyatla.kelpjava.loss.LossFunction.Result;

    //ネットワークの訓練を実行するクラス
    //主にArray->NdArrayの型変換を担う
    public class Trainer
    {
		public double train(FunctionStack i_functionStack, NdArray i_input, NdArray i_teach,LossFunction i_lossfunction) {
			return this.train(i_functionStack, i_input,i_teach,i_lossfunction,true);
		}



        //バッチで学習処理を行う
        public double train(FunctionStack i_functionStack, NdArray i_input, NdArray i_teach, LossFunction i_lossFunction, boolean i_isUpdate)
        {

            //Forwardのバッチを実行
            Result lossResult = i_lossFunction.evaluate(i_functionStack.forward(i_input),i_teach);

            //Backwardのバッチを実行
            i_functionStack.backward(lossResult.data);
            if (i_isUpdate)
            {
                i_functionStack.update();
            }
            return lossResult.loss;
        }

		public double batchTrain(FunctionStack i_functionStack, NdArray[] i_input, NdArray[] i_teach,LossFunction i_lossfunction) {
			return this.batchTrain(i_functionStack, i_input,i_teach,i_lossfunction,true);
		}
        public double batchTrain(FunctionStack i_functionStack, NdArray[] i_input, NdArray[] i_teach, LossFunction i_lossFunction, boolean i_isUpdate)
        {
            //Forwardのバッチを実行
        	LossFunction.Results lossResult = i_lossFunction.evaluate(i_functionStack.forward(i_input),i_teach);

            //Backwardのバッチを実行
            i_functionStack.backward(lossResult.data);

            if (i_isUpdate)
            {
                i_functionStack.update();
            }

            return lossResult.loss;
        }

        //精度測定
        public double accuracy(FunctionStack i_functionStack, NdArray[] i_x, NdArray[] i_y)
        {
            int matchCount = 0;

            NdArray[] forwardResult = i_functionStack.predict(i_x);

            for (int i = 0; i < i_x.length; i++)
            {
            	double max=JavaUtils.max(forwardResult[i].data);
                if (JavaUtils.indexOf(forwardResult[i].data,max) == i_y[i].data[0])
                {
                    matchCount++;
                }
            }

            return matchCount / (double)i_x.length;
        }





    }

