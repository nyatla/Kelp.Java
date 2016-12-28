package jp.nyatla.kelpjava.test;

import jp.nyatla.kelpjava.FunctionStack;
import jp.nyatla.kelpjava.Trainer;
import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.activations.ReLU;
import jp.nyatla.kelpjava.functions.connections.Linear;
import jp.nyatla.kelpjava.loss.MeanSquaredError;
import jp.nyatla.kelpjava.optimizers.Adam;
import jp.nyatla.kelpjava.optimizers.common.Optimizer;

/**
 * MLPによるXORの学習【回帰版】 ※精度が悪く何度か実行しないと望んだ結果を得られない
 * 
 */
public class Test2 {
	public static void main(String[] args) {
		// 訓練回数
		final int learningCount = 10000;

		// 訓練データ
		NdArray[] trainData = JavaUtils.createNdArray(new double[][] {
				{ 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 1.0 } });

		// 訓練データラベル
		NdArray[] trainLabel = JavaUtils.createNdArray(new double[][] {
				{ 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } });

		// ネットワークの構成を FunctionStack に書き連ねる
		FunctionStack nn = new FunctionStack(
			new Linear(2, 2, "l1 Linear"),
			new ReLU("l1 ReLU"),
			new Linear(2, 1, "l2 Linear"));

		// optimizerを宣言(今回はAdam)
		nn.setOptimizer(new Optimizer[]{new Adam()});

		Trainer trainer = new Trainer();
		MeanSquaredError loss_function = new MeanSquaredError();
		// 訓練ループ
		System.out.println("Training...");
		for (int i = 0; i < learningCount; i++) {
			// TrainerはOptimeserを省略すると更新を行わない
			trainer.train(nn, trainData[0], trainLabel[0], loss_function,false);
			trainer.train(nn, trainData[1], trainLabel[1], loss_function,false);
			trainer.train(nn, trainData[2], trainLabel[2], loss_function,false);
			trainer.train(nn, trainData[3], trainLabel[3], loss_function,false);

			// 訓練後に毎回更新を実行しなければ、ミニバッチとして更新できる
			nn.update();
		}

		// 訓練結果を表示
		System.out.println("Test Start...");
		for (NdArray val : trainData) {
			NdArray result = nn.predict(val);
			System.out.println(val.data[0] + " xor " + val.data[1] + " = "
					+ (result.data[0] > 0.5 ? 1 : 0) + " " + result);
		}
	}
}
