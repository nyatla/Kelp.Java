package jp.nyatla.kelpjava.test;

import java.io.IOException;


import jp.nyatla.kelpjava.FunctionStack;
import jp.nyatla.kelpjava.IOptimizer;
import jp.nyatla.kelpjava.Trainer;
import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.activations.Sigmoid;
import jp.nyatla.kelpjava.functions.connections.Linear;
import jp.nyatla.kelpjava.loss.SoftmaxCrossEntropy;
import jp.nyatla.kelpjava.optimizers.MomentumSGD;

//MLPによるXORの学習
class Test1 {

	public static void main(String[] i_args) {
		// 訓練回数
		final int learningCount = 10000;

		// 訓練データ
		NdArray[] trainData =JavaUtils.createNdArray(new double[][]{
			{0,0},
			{1,0},
			{0,1},
			{1,1},
		});

		// 訓練データラベル
		NdArray[] trainLabel = JavaUtils.createNdArray(new double[][]{
				{ 0.0 },{ 1.0 },{ 1.0 },{ 0.0 }});
		

		// ネットワークの構成は FunctionStack に書き連ねる
		FunctionStack nn = new FunctionStack(
			new Linear(2, 2, "l1 Linear"),
			new Sigmoid("l1 Sigmoid"),
			new Linear(2, 2, "l2 Linear"));

		// optimizerを宣言
		IOptimizer[][] momentumSGD = { nn.InitOptimizers(new MomentumSGD()) };

		// 訓練ループ
		Trainer trainer = new Trainer();
		System.out.println("Training...");
		SoftmaxCrossEntropy loss_function = new SoftmaxCrossEntropy();
		for (int i = 0; i < learningCount; i++) {
			for (int j = 0; j < trainData.length; j++) {
				// 訓練実行時にロス関数を記述
				trainer.train(nn, trainData[j], trainLabel[j], loss_function,momentumSGD);
			}
		}

		// 訓練結果を表示
		System.out.println("Test Start...");
		for (NdArray input : trainData) {
			NdArray result = trainer.predict(nn, input);
			int resultIndex = JavaUtils.indexOf(result.data,JavaUtils.max(result.data));
			System.out.println(input.data[0] + " xor " + input.data[1] + " = " + resultIndex + " " + result);
		}
        //保存
		try {
			JavaUtils.writeToFile(nn,"test1.nn");
		} catch (IOException e1) {
			e1.printStackTrace();
			return;
		}
        FunctionStack nn2=null;
        
        try {
			nn2=JavaUtils.readFromFile("test1.nn");
		} catch (ClassNotFoundException | IOException e) {
			e.printStackTrace();
			return;
		}
		System.out.println("Test2 Start...");
		for (NdArray input : trainData) {
			NdArray result = trainer.predict(nn2, input);
			int resultIndex = JavaUtils.indexOf(result.data,JavaUtils.max(result.data));
			System.out.println(input.data[0] + " xor " + input.data[1] + " = " + resultIndex + " " + result);
		}

	}
}
