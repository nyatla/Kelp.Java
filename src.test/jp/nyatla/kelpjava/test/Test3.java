package jp.nyatla.kelpjava.test;

import jp.nyatla.kelpjava.FunctionStack;
import jp.nyatla.kelpjava.Optimizer;
import jp.nyatla.kelpjava.Trainer;
import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.activations.Tanh;
import jp.nyatla.kelpjava.functions.connections.Linear;
import jp.nyatla.kelpjava.loss.MeanSquaredError;
import jp.nyatla.kelpjava.optimizers.SGD;

/**
 * MLPによるSin関数の学習
 * 学習対象の周期を増やしたり、サンプリング数(N)を増やすとスコアが悪化するので、
 * 課題として挑戦してみると良いかもしれない
 */
public class Test3 {
	/** 学習回数*/
	final static int EPOCH = 1000;
	/** 一周期の分割数*/
	final static int N = 50;

	public static void main(String[] args) {
		double[][] trainData_array = new double[N][];
		double[][] trainLabel_array = new double[N][];
		for (int i = 0; i < N; i++) {
			// Sin波を一周期分用意
			double radian = -Math.PI + Math.PI * 2.0 * i / (N - 1);
			trainData_array[i] = new double[] { radian };
			trainLabel_array[i] = new double[] { Math.sin(radian) };
		}
		NdArray[] trainData = JavaUtils.createNdArray(trainData_array);
		NdArray[] trainLabel = JavaUtils.createNdArray(trainLabel_array);

		// ネットワークの構成を FunctionStack に書き連ねる
		FunctionStack nn = new FunctionStack(new Linear(1, 4, "l1 Linear"),
				new Tanh("l1 Tanh"), new Linear(4, 1, "l2 Linear"));

		// optimizerの宣言
		nn.setOptimizer(new Optimizer[]{new SGD()});
		Trainer trainer = new Trainer();
		MeanSquaredError lossfunction = new MeanSquaredError();
		// 訓練ループ
		for (int i = 0; i < EPOCH; i++) {
			// 誤差集計用
			double loss = 0.0;

			for (int j = 0; j < N; j++) {
				// ネットワークは訓練を実行すると戻り値に誤差が返ってくる
				loss += trainer.train(nn, trainData[j], trainLabel[j],
						lossfunction);
			}

			if (i % (EPOCH / 10) == 0) {
				System.out.println("loss:" + loss / N);
				System.out.println("");
			}
		}

		// 訓練結果を表示
		System.out.println("Test Start...");

		for (NdArray val : trainData) {
			System.out.println(val.data[0] + ":"
					+ trainer.predict(nn, val).data[0]);
		}
	}
}
