package jp.nyatla.kelpjava.test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import jp.nyatla.kelpjava.FunctionStack;
import jp.nyatla.kelpjava.Trainer;
import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.activations.ReLU;
import jp.nyatla.kelpjava.functions.connections.Linear;
import jp.nyatla.kelpjava.functions.normalization.BatchNormalization;
import jp.nyatla.kelpjava.io.mnist.MnistData;
import jp.nyatla.kelpjava.io.mnist.MnistData.DataSet;
import jp.nyatla.kelpjava.loss.LossFunction;
import jp.nyatla.kelpjava.loss.MeanSquaredError;
import jp.nyatla.kelpjava.loss.SoftmaxCrossEntropy;
import jp.nyatla.kelpjava.optimizers.Adam;

/**
 * Decoupled Neural Interfaces using Synthetic GradientsによるMNIST（手書き文字）の学習
 * http://ralo23.hatenablog.com/entry/2016/10/22/233405
 * 
 */
class Test11 {
	// ミニバッチの数
	// ミニバッチにC#標準のParallelを使用しているため、大きくし過ぎると遅くなるので注意
	final static int BATCH_DATA_COUNT = 200;

	// 一世代あたりの訓練回数
	final static int TRAIN_DATA_COUNT = 300; // = 60000 / 20

	// 性能評価時のデータ数
	final static int TEST_DATA_COUNT = 1000;

	public static void main(String[] args) throws IOException {
		// MNISTのデータを用意する
		System.out.println("MNIST Data Loading...");
		MnistData train = new MnistData(new File("data/mnist/train-images.idx3-ubyte"), new File("data/mnist/train-labels.idx1-ubyte"));
		MnistData teach = new MnistData(new File("data/mnist/t10k-images.idx3-ubyte"), new File("data/mnist/t10k-labels.idx1-ubyte"));


		System.out.println("Training Start...");

		// ネットワークの構成を FunctionStack に書き連ねる
		FunctionStack Layer1 = new FunctionStack(new Linear(28 * 28, 256,
				"l1 Linear"), new BatchNormalization(256, "l1 Norm"), new ReLU(
				"l1 ReLU"));

		FunctionStack Layer2 = new FunctionStack(new Linear(256, 256,
				"l2 Linear"), new BatchNormalization(256, "l2 Norm"), new ReLU(
				"l2 ReLU"));

		FunctionStack Layer3 = new FunctionStack(new Linear(256, 256,
				"l3 Linear"), new BatchNormalization(256, "l3 Norm"), new ReLU(
				"l3 ReLU"));

		FunctionStack Layer4 = new FunctionStack(new Linear(256, 10,
				"l4 Linear"));

		// FunctionStack自身もFunctionとして積み上げられる
		FunctionStack nn = new FunctionStack(Layer1, Layer2, Layer3, Layer4);

		FunctionStack DNI1 = new FunctionStack(new Linear(256, 1024,
				"DNI1 Linear1"), new BatchNormalization(1024, "DNI1 Nrom1"),
				new ReLU("DNI1 ReLU1"), new Linear(1024, 1024, "DNI1 Linear2"),
				new BatchNormalization(1024, "DNI1 Nrom2"), new ReLU(
						"DNI1 ReLU2"), new Linear(1024, 256, false,
						new NdArray(new double[1024 * 256]), null,
						"DNI1 Linear3"));

		FunctionStack DNI2 = new FunctionStack(new Linear(256, 1024,
				"DNI2 Linear1"), new BatchNormalization(1024, "DNI2 Nrom1"),
				new ReLU("DNI2 ReLU1"), new Linear(1024, 1024, "DNI2 Linear2"),
				new BatchNormalization(1024, "DNI2 Nrom2"), new ReLU(
						"DNI2 ReLU2"), new Linear(1024, 256, false,
						new NdArray(new double[1024 * 256]), null,
						"DNI2 Linear3"));

		FunctionStack DNI3 = new FunctionStack(new Linear(256, 1024,
				"DNI3 Linear1"), new BatchNormalization(1024, "DNI3 Nrom1"),
				new ReLU("DNI3 ReLU1"), new Linear(1024, 1024, "DNI3 Linear2"),
				new BatchNormalization(1024, "DNI3 Nrom2"), new ReLU(
						"DNI3 ReLU2"), new Linear(1024, 256, false,
						new NdArray(new double[1024 * 256]), null,
						"DNI3 Linear3"));

		// optimizerを宣言
		Layer1.setOptimizer(new Adam());
		Layer2.setOptimizer(new Adam());
		Layer3.setOptimizer(new Adam());
		Layer4.setOptimizer(new Adam());

		DNI1.setOptimizer(new Adam());
		DNI2.setOptimizer(new Adam());
		DNI3.setOptimizer(new Adam());

		// 三世代学習
		for (int epoch = 0; epoch < 20; epoch++) {
			System.out.println("epoch " + (epoch + 1));

			// 全体での誤差を集計
			List<Double> totalLoss = new ArrayList<Double>();

			List<Double> DNI1totalLoss = new ArrayList<Double>();

			List<Double> DNI2totalLoss = new ArrayList<Double>();

			List<Double> DNI3totalLoss = new ArrayList<Double>();
			Trainer trainer = new Trainer();
			// 何回バッチを実行するか
			for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++) {
				// 訓練データからランダムにデータを取得
				DataSet datasetX = train.getRandomDataSet(BATCH_DATA_COUNT);

				// 第一層を実行
				NdArray[] layer1ForwardResult = Layer1.forward(datasetX.image);

				// 第一層の傾きを取得
				NdArray[] DNI1Result = DNI1.forward(layer1ForwardResult);

				// 第一層を更新
				Layer1.backward(DNI1Result);
				Layer1.update();

				// 第二層を実行
				NdArray[] layer2ForwardResult = Layer2
						.forward(layer1ForwardResult);

				// 第二層の傾きを取得
				NdArray[] DNI2Result = DNI2.forward(layer2ForwardResult);

				// 第二層を更新
				NdArray[] layer2BackwardResult = Layer2.backward(DNI2Result);
				Layer2.update();

				// 第一層用のDNIの学習を実行
				LossFunction.Results DNI1loss = new MeanSquaredError()
						.evaluate(DNI1Result, layer2BackwardResult);
				NdArray[] DNI1lossResult = DNI1loss.data;
				DNI1.backward(DNI1lossResult);
				DNI1.update();
				DNI1totalLoss.add(DNI1loss.loss);

				// 第二層を実行
				NdArray[] layer3ForwardResult = Layer3
						.forward(layer2ForwardResult);

				// 第三層の傾きを取得
				NdArray[] DNI3Result = DNI3.forward(layer3ForwardResult);

				// 第三層を更新
				NdArray[] layer3BackwardResult = Layer3.backward(DNI3Result);
				Layer3.update();

				// 第二層用のDNIの学習を実行
				LossFunction.Results DNI2loss = new MeanSquaredError()
						.evaluate(DNI2Result, layer3BackwardResult);
				NdArray[] DNI2lossResult = DNI2loss.data;
				DNI2.backward(DNI2lossResult);
				DNI2.update();
				DNI2totalLoss.add(DNI2loss.loss);

				// 第四層を実行
				NdArray[] layer4ForwardResult = Layer4
						.forward(layer3ForwardResult);

				// 第四層の傾きを取得
				LossFunction.Results sumLoss = new SoftmaxCrossEntropy()
						.evaluate(layer4ForwardResult, datasetX.label);
				NdArray[] lossResult = sumLoss.data;

				// 第四層を更新
				NdArray[] layer4BackwardResult = Layer4.backward(lossResult);
				Layer4.update();
				totalLoss.add(sumLoss.loss);

				// 第三層用のDNIの学習を実行
				LossFunction.Results DNI3loss = new MeanSquaredError()
						.evaluate(DNI3Result, layer4BackwardResult);
				NdArray[] DNI3lossResult = DNI3loss.data;
				DNI3.backward(DNI3lossResult);
				DNI3.update();
				DNI3totalLoss.add(DNI3loss.loss);

				System.out.println("\nbatch count " + i + "/"
						+ TRAIN_DATA_COUNT);
				// 結果出力
				System.out.println("total loss "
						+ JavaUtils.average(totalLoss.toArray(new Double[0])));
				System.out.println("local loss " + sumLoss.loss);

				System.out.println("\nDNI1 total loss "
						+ JavaUtils.average(DNI1totalLoss
								.toArray(new Double[0])));
				System.out.println("DNI2 total loss "
						+ JavaUtils.average(DNI2totalLoss
								.toArray(new Double[0])));
				System.out.println("DNI3 total loss "
						+ JavaUtils.average(DNI3totalLoss
								.toArray(new Double[0])));

				System.out.println("\nDNI1 local loss " + DNI1loss.loss);
				System.out.println("DNI2 local loss " + DNI2loss.loss);
				System.out.println("DNI3 local loss " + DNI3loss.loss);

				// 20回バッチを動かしたら精度をテストする
				if (i % 20 == 0) {
					System.out.println("\nTesting...");

					// テストデータからランダムにデータを取得
					DataSet datasetY = teach.getRandomDataSet(TEST_DATA_COUNT);

					// テストを実行
					double accuracy = trainer.accuracy(nn, datasetY.image,
							datasetY.label);
					System.out.println("accuracy " + accuracy);
				}
			}
		}
	}
}
