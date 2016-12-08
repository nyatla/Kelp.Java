package jp.nyatla.kelpjava;

import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.loss.LossFunctions;
import jp.nyatla.kelpjava.loss.LossFunctions.Results;
import jp.nyatla.kelpjava.loss.MeanSquaredError;
import jp.nyatla.kelpjava.loss.SingleLossFunction;
import jp.nyatla.kelpjava.loss.SingleLossFunction.Result;

    //ネットワークの訓練を実行するクラス
    //主にArray->NdArrayの型変換を担う
    public class Trainer
    {
        //ロス関数のデリゲート宣言
//        public delegate NdArray[] LossFunction(NdArray[] input, NdArray[] teachSignal, out double loss);
//        public delegate NdArray SingleLossFunction(NdArray input, NdArray teachSignal, out double loss);

//        public static NdArray[] Forward(Function function, Array[] input)
//        {
//            return function.Forward(NdArray.FromArray(input));
//        }
//
//        public static NdArray Forward(Function function, Array input)
//        {
//            return function.Forward(NdArray.FromArray(input));
//        }
//
//        public static NdArray[] Forward(Function function, Array[] input, Array[] teach, LossFunction lossFunction, out double sumLoss)
//        {
//            return lossFunction(function.Forward(NdArray.FromArray(input)), NdArray.FromArray(teach), out sumLoss);
//        }
//
//        public static NdArray Forward(Function function, Array input, Array teach, SingleLossFunction lossFunction, out double sumLoss)
//        {
//            return lossFunction(function.Forward(NdArray.FromArray(input)), NdArray.FromArray(teach), out sumLoss);
//        }
//
//        public static NdArray[] Backward(Function function, Array[] input)
//        {
//            return function.Backward(NdArray.FromArray(input));
//        }
//
//        public static NdArray Backward(Function function, Array input)
//        {
//            return function.Backward(NdArray.FromArray(input));
//        }

		public double train(FunctionStack i_functionStack, NdArray i_input, NdArray i_teach,MeanSquaredError i_lossfunction, IOptimizer[] i_optimizers) {
            //Forwardのバッチを実行
            Result lossResult = i_lossfunction.evaluate(i_functionStack.forward(i_input),i_teach);

            //Backwardのバッチを実行
            i_functionStack.backward(lossResult.data);
            if (i_optimizers!=null && i_optimizers.length>0)
            {
                i_functionStack.update(i_optimizers);
            }
            return lossResult.loss;
		}
        public double train(FunctionStack i_functionStack, NdArray i_input, NdArray i_teach, SingleLossFunction i_lossFunction)
        {
        	return this.train(i_functionStack,i_input,i_teach,i_lossFunction,null);
        }


        //バッチで学習処理を行う
        public double train(FunctionStack i_functionStack, NdArray i_input, NdArray i_teach, SingleLossFunction i_lossFunction,IOptimizer[][] i_optimizers)
        {

            //Forwardのバッチを実行
            Result lossResult = i_lossFunction.evaluate(i_functionStack.forward(i_input),i_teach);

            //Backwardのバッチを実行
            i_functionStack.backward(lossResult.data);
            if (i_optimizers!=null && i_optimizers.length>0)
            {
                i_functionStack.update(i_optimizers);
            }
            return lossResult.loss;
        }


        public double train(FunctionStack i_functionStack, NdArray[] i_input, NdArray[] i_teach, LossFunctions i_lossFunction, boolean i_isUpdate,IOptimizer[][] i_optimizers)
        {
            //Forwardのバッチを実行
            Results lossResult = i_lossFunction.evaluate(i_functionStack.forward(i_input),i_teach);

            //Backwardのバッチを実行
            i_functionStack.backward(lossResult.data);

            if (i_isUpdate)
            {
                i_functionStack.update(i_optimizers);
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

