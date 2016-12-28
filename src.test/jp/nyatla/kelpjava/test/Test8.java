package jp.nyatla.kelpjava.test;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

import jp.nyatla.kelpjava.FunctionStack;
import jp.nyatla.kelpjava.Trainer;
import jp.nyatla.kelpjava.common.JavaUtils;
import jp.nyatla.kelpjava.common.Mother;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.connections.LSTM;
import jp.nyatla.kelpjava.functions.connections.Linear;
import jp.nyatla.kelpjava.loss.LossFunction;
import jp.nyatla.kelpjava.loss.MeanSquaredError;
import jp.nyatla.kelpjava.optimizers.Adam;

/**
 * LSTMによるSin関数の学習（t の値から t+1 の値を予測する） 参考：
 * http://seiya-kumada.blogspot.jp/2016/07/lstm-chainer.html
 * 
 */

public class Test8 {
	final static int STEPS_PER_CYCLE = 50;
	final static int NUMBER_OF_CYCLES = 100;

	final static int TRAINING_EPOCHS = 1000;
	final static int MINI_BATCH_SIZE = 100;
	final static int LENGTH_OF_SEQUENCE = 100;

	final static int DISPLAY_EPOCH = 1;
	final static int PREDICTION_LENGTH = 75;

	public static void main(String[] args) {
		DataMaker dataMaker = new DataMaker(STEPS_PER_CYCLE, NUMBER_OF_CYCLES);
		NdArray trainData = dataMaker.Make();

		// ネットワークの構成は FunctionStack に書き連ねる
		FunctionStack model = new FunctionStack(new Linear(1, 5, "Linear l1"),
				new LSTM(5, 5, "LSTM l2"), new Linear(5, 1, "Linear l3"));

		// optimizerを宣言
		model.setOptimizer(new Adam());

		// 訓練ループ
		System.out.println("Training...");
		for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
			NdArray[] sequences = dataMaker.MakeMiniBatch(trainData,
					MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);

			double loss = ComputeLoss(model, sequences);

			model.update();

			model.resetState();

			if (epoch != 0 && epoch % DISPLAY_EPOCH == 0) {
				System.out.printf("[%d]training loss:\t%f\n", epoch, loss);
			}
		}

		System.out.println("Testing...");
		NdArray[] testSequences = dataMaker.MakeMiniBatch(trainData,
				MINI_BATCH_SIZE, LENGTH_OF_SEQUENCE);

		int sample_index = 45;
		predict(testSequences[sample_index], model, PREDICTION_LENGTH);
	}

	static double ComputeLoss(FunctionStack model, NdArray[] sequences) {
		// 全体での誤差を集計
		double[] totalLoss = new double[LENGTH_OF_SEQUENCE - 1];
		NdArray[] x = new NdArray[MINI_BATCH_SIZE];
		NdArray[] t = new NdArray[MINI_BATCH_SIZE];

		Stack<NdArray[]> backNdArrays = new Stack<NdArray[]>();

		for (int i = 0; i < LENGTH_OF_SEQUENCE - 1; i++) {
			for (int j = 0; j < MINI_BATCH_SIZE; j++) {
				x[j] = new NdArray(new double[] { sequences[j].data[i] });
				t[j] = new NdArray(new double[] { sequences[j].data[i + 1] });
			}

			LossFunction.Results r = new MeanSquaredError().evaluate(
					model.forward(x), t);
			backNdArrays.push(r.data);
			totalLoss[i] = (r.loss);
		}

		for (; backNdArrays.size() > 0;) {
			model.backward(backNdArrays.pop());
		}

		return JavaUtils.average(totalLoss);
	}

	static void predict(NdArray seq, FunctionStack model, int pre_length) {
		double[] pre_input_seq = new double[seq.length() / 4];
		if (pre_input_seq.length < 1) {
			pre_input_seq = new double[1];
		}
		System.arraycopy(seq.data, 0, pre_input_seq, 0, pre_input_seq.length);

		List<Double> input_seq = new ArrayList<Double>();
		for (int i = 0; i < pre_input_seq.length; i++) {
			input_seq.add(pre_input_seq[i]);
		}

		List<Double> output_seq = new ArrayList<Double>();
		output_seq.add(input_seq.get(input_seq.size() - 1));

		for (int i = 0; i < pre_length; i++) {
			double future = predict_sequence(model, input_seq);
			input_seq.remove(0);
			input_seq.add(future);
			output_seq.add(future);
		}

		for (int i = 0; i < output_seq.size(); i++) {
			System.out.println(output_seq.get(i));
		}

		System.out.println(seq);
	}

	static double predict_sequence(FunctionStack model, List<Double> input_seq) {
		model.resetState();

		NdArray result = NdArray.zeros(1);
		for (int i = 0; i < input_seq.size(); i++) {
			result = model.predict(new NdArray(
					new double[] { input_seq.get(i) }));
		}

		return result.data[0];
	}

	static class DataMaker {
		private final int stepsPerCycle;
		private final int numberOfCycles;

		public DataMaker(int stepsPerCycle, int numberOfCycles) {
			this.stepsPerCycle = stepsPerCycle;
			this.numberOfCycles = numberOfCycles;
		}

		public NdArray Make() {
			NdArray result = NdArray.zeros(this.stepsPerCycle
					* this.numberOfCycles);

			for (int i = 0; i < this.numberOfCycles; i++) {
				for (int j = 0; j < this.stepsPerCycle; j++) {
					result.data[j + i * this.stepsPerCycle] = Math.sin(j * 2
							* Math.PI / this.stepsPerCycle);
				}
			}

			return result;
		}

		public NdArray[] MakeMiniBatch(NdArray baseFreq, int miniBatchSize,
				int lengthOfSequence) {
			NdArray[] result = new NdArray[miniBatchSize];

			for (int j = 0; j < result.length; j++) {
				result[j] = NdArray.zeros(lengthOfSequence);

				int index = Mother.Dice.nextInt(baseFreq.length()
						- lengthOfSequence);
				for (int i = 0; i < lengthOfSequence; i++) {
					result[j].data[i] = baseFreq.data[index + i];
				}

			}

			return result;
		}
	}
}
