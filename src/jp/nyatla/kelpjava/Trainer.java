package jp.nyatla.kelpjava;

import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.loss.LossFunctions;
import jp.nyatla.kelpjava.loss.LossFunctions.Results;
import jp.nyatla.kelpjava.loss.SingleLossFunction;
import jp.nyatla.kelpjava.loss.SingleLossFunction.Result;

    //ネットワークの訓練を実行するクラス
    //主にArray->NdArrayの型変換を担う
    public class Trainer
    {
		public double train(FunctionStack i_functionStack, NdArray i_input, NdArray i_teach,SingleLossFunction i_lossfunction) {
			return this.train(i_functionStack, i_input,i_teach,i_lossfunction,true);
		}



        //バッチで学習処理を行う
        public double train(FunctionStack i_functionStack, NdArray i_input, NdArray i_teach, SingleLossFunction i_lossFunction, boolean i_isUpdate)
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

		public double train(FunctionStack i_functionStack, NdArray[] i_input, NdArray[] i_teach,LossFunctions i_lossfunction) {
			return this.train(i_functionStack, i_input,i_teach,i_lossfunction,true);
		}
        public double train(FunctionStack i_functionStack, NdArray[] i_input, NdArray[] i_teach, LossFunctions i_lossFunction, boolean i_isUpdate)
        {
            //Forwardのバッチを実行
            Results lossResult = i_lossFunction.evaluate(i_functionStack.forward(i_input),i_teach);

            //Backwardのバッチを実行
            i_functionStack.backward(lossResult.data);

            if (i_isUpdate)
            {
                i_functionStack.update();
            }

            return lossResult.loss;
        }

        //精度測定
        public double accuracy(FunctionStack i_functionStack, NdArray[] i_x, int[][] i_y)
        {
            int matchCount = 0;

            NdArray[] forwardResult = i_functionStack.predict(i_x);

            for (int i = 0; i < i_x.length; i++)
            {
            	double max=JavaUtils.max(forwardResult[i].data);
                if (JavaUtils.indexOf(forwardResult[i].data,max) == i_y[i][0])
                {
                    matchCount++;
                }
            }

            return matchCount / (double)i_x.length;
        }

        //予想を実行する（外部からの使用を想定してArrayが引数
        public NdArray[] predict(FunctionStack functionStack, NdArray[] i_input)
        {
            NdArray[] ndArrays = new NdArray[i_input.length];

            for (int i = 0; i < ndArrays.length; i++)
            {
                ndArrays[i] = i_input[i];
            }

            return functionStack.predict(ndArrays);
        }

        //予想を実行する[非バッチ]（外部からの使用を想定してArrayが引数
        public NdArray predict(FunctionStack functionStack, NdArray i_input)
        {
            return functionStack.predict(i_input);
        }




    }

