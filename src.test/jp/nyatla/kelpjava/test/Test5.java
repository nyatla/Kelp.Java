package jp.nyatla.kelpjava.test;

import jp.nyatla.kelpjava.FunctionStack;
import jp.nyatla.kelpjava.Trainer;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.activations.ReLU;
import jp.nyatla.kelpjava.functions.connections.Convolution2D;
import jp.nyatla.kelpjava.functions.connections.Linear;
import jp.nyatla.kelpjava.functions.poolings.MaxPooling;
import jp.nyatla.kelpjava.loss.MeanSquaredError;
import jp.nyatla.kelpjava.optimizers.SGD;

//エクセルCNNの再現
class Test5 {
	public static void main(String[] args) {
		// 各初期値を記述
		NdArray initial_W1 = new NdArray(new double[][][][] {
				{ { { 1.0, 0.5, 0.0 }, { 0.5, 0.0, -0.5 },	{ 0.0, -0.5, -1.0 } } },
				{ { { 0.0, -0.1, 0.1 }, { -0.3, 0.4, 0.7 },	{ 0.5, -0.2, 0.2 } } } });
		NdArray initial_b1 = new NdArray(new double[] { 0.5, 1.0 });

		NdArray initial_W2 = new NdArray(new double[][][][] {
				{ { { -0.1, 0.6 }, { 0.3, -0.9 } },	{ { 0.7, 0.9 }, { -0.2, -0.3 } } },
				{ { { -0.6, -0.1 }, { 0.3, 0.3 } },	{ { -0.5, 0.8 }, { 0.9, 0.1 } } } });
		NdArray initial_b2 = new NdArray(new double[] { 0.1, 0.9 });

		NdArray initial_W3 = new NdArray(new double[][] {
				{ 0.5, 0.3, 0.4, 0.2, 0.6, 0.1, 0.4, 0.3 },
				{ 0.6, 0.4, 0.9, 0.1, 0.5, 0.2, 0.3, 0.4 } });
		NdArray initial_b3 = new NdArray(new double[] { 0.01, 0.02 });

		NdArray initial_W4 = new NdArray(new double[][] { { 0.8, 0.2 },
				{ 0.4, 0.6 } });
		NdArray initial_b4 = new NdArray(new double[] { 0.02, 0.01 });

		// 入力データ
		NdArray x = new NdArray(
				new double[][][] { {
						{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.9, 0.2, 0.0, 0.0,	0.0, 0.0 },
						{ 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.9, 0.1, 0.0, 0.0,	0.0, 0.0 },
						{ 0.0, 0.0, 0.0, 0.1, 0.8, 0.5, 0.8, 0.1, 0.0, 0.0,	0.0, 0.0 },
						{ 0.0, 0.0, 0.0, 0.3, 0.3, 0.1, 0.7, 0.2, 0.0, 0.0,	0.0, 0.0 },
						{ 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0,	0.0, 0.0 },
						{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.1, 0.0, 0.0,	0.0, 0.0 },
						{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 0.1, 0.0, 0.0,	0.0, 0.0 },
						{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.4, 0.1, 0.0, 0.0,	0.0, 0.0 },
						{ 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.3, 0.0, 0.0, 0.0,	0.0, 0.0 },
						{ 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.2, 0.0, 0.0, 0.0,	0.0, 0.0 },
						{ 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0,	0.0, 0.0 },
						{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0,	0.0, 0.0 } } });

		// 教師信号
		NdArray t = new NdArray(new double[] { 0.0, 1.0 });

		// 層の中身をチェックしたい場合は、層単体でインスタンスを持つ
		Convolution2D l2 = new Convolution2D(1, 2, 3, 1, 0, false, initial_W1,initial_b1, "l2 Conv2D");

		// ネットワークの構成を FunctionStack に書き連ねる
		FunctionStack nn = new FunctionStack(
				l2,
				new ReLU("l2 ReLU"),
				new MaxPooling(2, 2, "l2 Pooling"),
				new Convolution2D(2, 2, 2, 1, 0, false, initial_W2, initial_b2,"l3 Conv2D"),
				new ReLU("l3 ReLU"),
				new MaxPooling(2, 2,"l3 Pooling"),
				new Linear(8, 2, false, initial_W3,initial_b3, "l4 Linear"),
				new ReLU("l4 ReLU"),
				new Linear(2, 2, false, initial_W4, initial_b4, "l5 Linear"));

		// optimizerの宣言を省略するとデフォルトのSGD(0.1)が使用される
		nn.setOptimizer(new SGD());

		Trainer trainer = new Trainer();

		// 訓練を実施
		trainer.train(nn, x, t, new MeanSquaredError(), false);

		// Updateを実行するとgradが消費されてしまうため値を先に出力
		System.out.println("gw1");
		System.out.println(l2.gW);

		System.out.println("gb1");
		System.out.println(l2.gb);

		// 更新
		nn.update();

		System.out.println("w1");
		System.out.println(l2.W);

		System.out.println("b1");
		System.out.println(l2.b);
	}
}
