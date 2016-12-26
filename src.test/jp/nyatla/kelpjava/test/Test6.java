package jp.nyatla.kelpjava.test;

import java.io.File;
import java.io.IOException;

import jp.nyatla.kelpjava.FunctionStack;
import jp.nyatla.kelpjava.Trainer;
import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.functions.activations.ReLU;
import jp.nyatla.kelpjava.functions.connections.Convolution2D;
import jp.nyatla.kelpjava.functions.connections.Linear;
import jp.nyatla.kelpjava.functions.noise.Dropout;
import jp.nyatla.kelpjava.functions.poolings.MaxPooling;
import jp.nyatla.kelpjava.io.mnist.MnistData;
import jp.nyatla.kelpjava.io.mnist.MnistData.DataSet;
import jp.nyatla.kelpjava.loss.SoftmaxCrossEntropy;
import jp.nyatla.kelpjava.optimizers.Adam;

//5層CNNによるMNIST（手書き文字）の学習
//Test4と違うのはネットワークの構成とOptimizerだけです
class Test6 {
	// ミニバッチの数
	// ミニバッチにC#標準のParallelを使用しているため、大きくし過ぎると遅くなるので注意
	final static int BATCH_DATA_COUNT = 20;

	// 一世代あたりの訓練回数
	final static int TRAIN_DATA_COUNT = 3000; // = 60000 / 20

	// 性能評価時のデータ数
	final static int TEACH_DATA_COUNT = 200;

	public static void main(String[] args) throws IOException {

		// MNISTのデータを用意する
		System.out.println("MNIST Data Loading...");
		MnistData train = new MnistData(new File(
				"data/mnist/train-images.idx3-ubyte"), new File(
				"data/mnist/train-labels.idx1-ubyte"));
		MnistData teach = new MnistData(new File(
				"data/mnist/t10k-images.idx3-ubyte"), new File(
				"data/mnist/t10k-labels.idx1-ubyte"));

		// ネットワークの構成を FunctionStack に書き連ねる
		FunctionStack nn = new FunctionStack(new Convolution2D(1, 32, 5, 2,
				"l1 Conv2D"), new ReLU("l1 ReLU"), new MaxPooling(2, 2,
				"l1 MaxPooling"), new Convolution2D(32, 64, 5, 2, "l2 Conv2D"),
				new ReLU("l2 ReLU"), new MaxPooling(2, 2, "l2 MaxPooling"),
				new Linear(7 * 7 * 64, 1024, "l3 Linear"), new Dropout(
						"l3 DropOut"), new ReLU("l3 ReLU"), new Linear(1024,
						10, "l4 Linear"));

		// optimizerを宣言
		nn.setOptimizer(new Adam());
		Trainer trainer = new Trainer();
		System.out.println("Training Start...");

		// 三世代学習
		for (int epoch = 1; epoch < 3; epoch++) {
			System.out.println("epoch " + epoch);

			// 全体での誤差を集計
			double[] totalLoss = new double[TRAIN_DATA_COUNT - 1];

			// 何回バッチを実行するか
			for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++) {
				long sw_start = System.currentTimeMillis();

				System.out.println("\nbatch count " + i + "/"
						+ TRAIN_DATA_COUNT);

				// 訓練データからランダムにデータを取得
				DataSet datasetX = train.getRandomDataSet(BATCH_DATA_COUNT);

				// バッチ学習を並列実行する
				double sumLoss = trainer.batchTrain(nn, datasetX.image,
						datasetX.label, new SoftmaxCrossEntropy());
				totalLoss[i - 1] = sumLoss;

				// 結果出力
				System.out.println("total loss "
						+ JavaUtils.average(totalLoss, i));
				System.out.println("local loss " + sumLoss);

				System.out.println("time "
						+ (System.currentTimeMillis() - sw_start));

				// 20回バッチを動かしたら精度をテストする
				if (i % 20 == 0) {
					System.out.println("\nTesting...");

					// テストデータからランダムにデータを取得
					DataSet datasetY = teach.getRandomDataSet(TEACH_DATA_COUNT);

					// テストを実行
					double accuracy = trainer.accuracy(nn, datasetY.image,
							datasetY.label);
					System.out.println("accuracy " + accuracy);
				}
			}
		}
	}
}
