package jp.nyatla.kelpjava.test;

import java.io.File;
import java.io.IOException;
import java.util.Stack;

import jp.nyatla.kelpjava.FunctionStack;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.activations.Softmax;
import jp.nyatla.kelpjava.functions.activations.Tanh;
import jp.nyatla.kelpjava.functions.connections.EmbedID;
import jp.nyatla.kelpjava.functions.connections.Linear;
import jp.nyatla.kelpjava.io.vocabulary.IndexedTextData;
import jp.nyatla.kelpjava.io.vocabulary.VocabularyLines;
import jp.nyatla.kelpjava.io.vocabulary.VocabularyText;
import jp.nyatla.kelpjava.loss.LossFunction;
import jp.nyatla.kelpjava.loss.SoftmaxCrossEntropy;
import jp.nyatla.kelpjava.optimizers.Adam;

/**
 * SimpleなRNNによるRNNLM 『Chainerによる実践深層学習』より（ISBN 978-4-274-21934-4）
 * 
 */
public class Test9 {
	final private static int TRAINING_EPOCHS = 5;
	final private static int N_UNITS = 100;

	public static void main(String[] args) throws IOException {
		System.out.println("Build Vocabulary.");

		VocabularyLines trainText = new VocabularyLines(new File("data/ptb/ptb.train.txt"));
		VocabularyLines testText = new VocabularyLines(new File("data/ptb/ptb.test.txt"));

		IndexedTextData textdata = new IndexedTextData();
		textdata.add(trainText.lines);
		textdata.add(testText.lines);
		int eos_id = textdata.getId(VocabularyText.EOS);
		int[][] trainData = textdata.getTextIds(trainText.lines);
		int[][] testData = textdata.getTextIds(testText.lines);

		int nVocab = textdata.getLength();

		System.out.println("Done.");

		System.out.println("Network Initilizing.");
		FunctionStack model = new FunctionStack(
				new EmbedID(nVocab, N_UNITS,"l1 EmbedID"),
				new Linear(N_UNITS, N_UNITS, "l2 Linear"),
				new Tanh("l2 Tanh"), new Linear(N_UNITS, nVocab, "l3 Linear"),
				new Softmax("l3 Sonftmax"));

		model.setOptimizer(new Adam());

		System.out.println("Train Start.");
		SoftmaxCrossEntropy softmaxCrossEntropy = new SoftmaxCrossEntropy();
		for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
			NdArray h = NdArray.zeros(N_UNITS);
			for (int pos = 0; pos < trainData.length; pos++) {

				int[] s = trainData[pos];
				double accumloss = 0;
				Stack<NdArray> tmp = new Stack<NdArray>();

				for (int i = 0; i < s.length; i++) {
					// 最後の1文字は改行コード
					int tx = (i == s.length - 1 ? eos_id : s[i + 1]);
					// l1 Linear
					NdArray xK = model.functions[0].forward(new NdArray(
							new double[] { s[i] }));

					// l2 Linear
					NdArray l2 = model.functions[1].forward(h);
					for (int j = 0; j < xK.length(); j++) {
						xK.data[j] += l2.data[j];
					}

					// l2 Tanh
					h = model.functions[2].forward(xK);

					// l3 Linear
					NdArray h2 = model.functions[3].forward(h);

					LossFunction.Result loss = softmaxCrossEntropy.evaluate(h2,
							NdArray.fromArray(new double[] { tx }));
					tmp.push(loss.data);
					accumloss += loss.loss;
				}

				System.out.println(accumloss);

				for (int i = 0; i < s.length; i++) {
					NdArray g = model.functions[3].backward(tmp.pop());
					g = model.functions[2].backward(g);
					g = model.functions[1].backward(g);
					model.functions[0].backward(g);
				}

				model.update();
				// }

				if (pos % 100 == 0) {
					System.out.println(pos + "/" + trainData.length
							+ " finished");
				}
			}
		}

		System.out.println("Test Start.");

		double sum = 0.0;
		int wnum = 0;
		boolean unkWord = false;

		for (int pos = 0; pos < 1000; pos++) {
			int[] ts = testData[pos];

			if (!unkWord) {
				System.out.println("pos" + pos);
				System.out.println("tsLen" + ts.length);
				System.out.println("sum" + sum);
				System.out.println("wnum" + wnum);

				sum += CalPs(model, ts);
				wnum += ts.length - 1;
			} else {
				unkWord = false;
			}
		}

		System.out.println(Math.pow(2.0, sum / wnum));
	}

	static double CalPs(FunctionStack model, int[] s) {
		double sum = 0.0;

		NdArray h = NdArray.zeros(N_UNITS);

		for (int i = 1; i < s.length; i++) {
			// l1 Linear
			NdArray xK = model.functions[0].forward(new NdArray(
					new double[] { s[i] }));

			// l2 Linear
			NdArray l2 = model.functions[1].forward(h);
			for (int j = 0; j < xK.length(); j++) {
				xK.data[j] += l2.data[j];
			}

			// l2 Tanh
			h = model.functions[2].forward(xK);

			// l3 Softmax(l3 Linear)
			NdArray yv = model.functions[4].forward(model.functions[3]
					.forward(h));
			double pi = yv.data[s[i - 1]];
			sum -= Math.log(pi) / Math.log(2);
		}

		return sum;
	}
}
