package jp.nyatla.kelpjava.test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

import jp.nyatla.kelpjava.FunctionStack;
import jp.nyatla.kelpjava.Optimizer;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.connections.EmbedID;
import jp.nyatla.kelpjava.functions.connections.LSTM;
import jp.nyatla.kelpjava.functions.connections.Linear;
import jp.nyatla.kelpjava.functions.noise.Dropout;
import jp.nyatla.kelpjava.io.vocabulary.IndexedTextData;
import jp.nyatla.kelpjava.io.vocabulary.VocabularyText;
import jp.nyatla.kelpjava.loss.LossFunction;
import jp.nyatla.kelpjava.loss.LossFunction.Results;
import jp.nyatla.kelpjava.loss.SoftmaxCrossEntropy;
import jp.nyatla.kelpjava.optimizers.GradientClipping;
import jp.nyatla.kelpjava.optimizers.SGD;

/**
 * ChainerのRNNサンプルを再現 https://github.com/pfnet/chainer/tree/master/examples/ptb
 */
class Test10 {
	final static int N_EPOCH = 39;
	final static int N_UNITS = 650;
	final static int BATCH_SIZE = 20;
	final static int BPROP_LEN = 35;
	final static int GRAD_CLIP = 5;

	public static void main(String[] args) throws IOException {
		System.out.println("Build Vocabulary.");

		VocabularyText trainText = new VocabularyText(new File("data/ptb/ptb.train.txt"));
		VocabularyText validText = new VocabularyText(new File("data/ptb/ptb.valid.txt"));
		VocabularyText testText = new VocabularyText(new File("data/ptb/ptb.test.txt"));

		IndexedTextData textdata = new IndexedTextData();
		textdata.add(trainText.text);
		textdata.add(validText.text);
		textdata.add(testText.text);
		int[] trainData = textdata.getTextIds(trainText.text);
		int[] testData = textdata.getTextIds(testText.text);
		int[] validData=textdata.getTextIds(validText.text);

		int nVocab = textdata.getLength();

		System.out.println("Network Initilizing.");
		FunctionStack model = new FunctionStack(
				new EmbedID(nVocab, N_UNITS,"l1 EmbedID"),
				new Dropout(),
				new LSTM(N_UNITS, N_UNITS,"l2 LSTM"),
				new Dropout(),
				new LSTM(N_UNITS, N_UNITS, "l3 LSTM"),
				new Dropout(),
				new Linear(N_UNITS, nVocab, "l4 Linear"));

		// 与えられたthresholdで頭打ちではなく、全パラメータのL2Normからレートを取り補正を行う
		GradientClipping gradientClipping = new GradientClipping(GRAD_CLIP);
		SGD sgd = new SGD(1.0);
		model.setOptimizer(new Optimizer[] { gradientClipping, sgd });

		double wholeLen = trainData.length;
		int jump = (int) Math.floor(wholeLen / BATCH_SIZE);
		int epoch = 0;

		Stack<NdArray[]> backNdArrays = new Stack<NdArray[]>();

		System.out.println("Train Start.");

		for (int i = 0; i < jump * N_EPOCH; i++) {
			NdArray[] x = new NdArray[BATCH_SIZE];
			NdArray[] t = new NdArray[BATCH_SIZE];

			for (int j = 0; j < BATCH_SIZE; j++) {
				x[j] = new NdArray(new double[]{trainData[(int) ((jump* j + i) % wholeLen)]});
				t[j] = new NdArray(new double[]{trainData[(int) ((jump* j + i + 1) % wholeLen)]});
			}

			SoftmaxCrossEntropy softmaxCrossEntropy = new SoftmaxCrossEntropy();
			Results sumLoss = softmaxCrossEntropy.evaluate(model.forward(x), t);

			backNdArrays.push(sumLoss.data);
			System.out.printf("[%d/%d] Loss:%f\n", i + 1, jump, sumLoss.loss);

			// Run truncated BPTT
			if ((i + 1) % BPROP_LEN == 0) {
				for (int j = 0; backNdArrays.size() > 0; j++) {
					System.out.println("backward" + backNdArrays.size());
					model.backward(backNdArrays.pop());
				}

				model.update();
				model.resetState();
			}

			if ((i + 1) % jump == 0) {
				epoch++;
				System.out.println("evaluate");
				System.out.printf("validation perplexity: %f\n",
						Evaluate(model, validData));

				if (epoch >= 6) {
					sgd.learningRate /= 1.2;
					System.out.println("learning rate =" + sgd.learningRate);
				}
			}
		}

		System.out.println("test start");
		double testPerp = Evaluate(model, testData);
		System.out.println("test perplexity:" + testPerp);
	}

	static double Evaluate(FunctionStack model, int[] dataset) {
		FunctionStack predictModel = (FunctionStack) model.deepCopy();
		predictModel.resetState();

		List<Double> totalLoss = new ArrayList<Double>();

		for (int i = 0; i < dataset.length - 1; i++) {
			NdArray[] x = new NdArray[BATCH_SIZE];
			NdArray[] t = new NdArray[BATCH_SIZE];

			for (int j = 0; j < BATCH_SIZE; j++) {
				x[j] = NdArray.fromArray(new double[] { dataset[j + i] });
				t[j] = NdArray.fromArray(new double[] { dataset[j + i + 1] });
			}

			LossFunction.Results sumLoss;
			sumLoss = new SoftmaxCrossEntropy().evaluate(
					predictModel.forward(x), t);
			totalLoss.add(sumLoss.loss);
		}
		double total_loss = 0;
		for (Double i : totalLoss) {
			total_loss += i;
		}

		// calc perplexity
		return Math.exp(total_loss / (totalLoss.size() - 1));
	}
}
