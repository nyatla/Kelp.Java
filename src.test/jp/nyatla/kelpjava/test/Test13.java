package jp.nyatla.kelpjava.test;

import jp.nyatla.kelpjava.common.Mother;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.connections.Deconvolution2D;
import jp.nyatla.kelpjava.loss.LossFunction;
import jp.nyatla.kelpjava.loss.MeanSquaredError;
import jp.nyatla.kelpjava.optimizers.SGD;

/**
 * ある学習済みフィルタで出力された画像を元に、そのフィルタと同等のフィルタを獲得する
 * コンソール版
 * 移植元 : http://qiita.com/samacoba/items/958c02f455ca5f3a475d
 *
 */
class Test13 {
	public static void main(String[] args)
	{
		NdArray w = new NdArray(MakeOneCore(15), new int[] { 1, 1, 15, 15 },false);
		// 目標とするフィルタを作成（実践であればココは不明な値となる）
		Deconvolution2D decon_core = new Deconvolution2D(1, 1, 15, 1, 7, false,w, null);

		Deconvolution2D model = new Deconvolution2D(1, 1, 15, 1, 7);

		SGD optimizer = new SGD(0.00005); // 大きいと発散する
		model.setOptimizer(optimizer);
		MeanSquaredError meanSquaredError = new MeanSquaredError();

		// ランダムに点が打たれた画像を生成
		NdArray img_p = getRandomImage(1, 128, 128);

		// 目標とするフィルタで学習用の画像を出力
		NdArray img_core = decon_core.forward(img_p);

		// 移植元では同じ教育画像で教育しているが、より実践に近い学習に変更
		for (int i = 0; i < 31; i++) {
			model.ClearGrads();

			// 未学習のフィルタで画像を出力
			NdArray img_y = model.forward(img_p);

			LossFunction.Result loss = meanSquaredError.evaluate(img_y,	img_core);

			model.backward(loss.data);

			model.update();

			System.out.println("epoch" + i + " : " + loss.loss);
		}
	}

	private static NdArray getRandomImage(int N, int img_w, int img_h) {
		// ランダムに0.1％の点を作る
		double[] img_p = new double[N * img_w * img_h];

		for (int i = 0; i < img_p.length; i++) {
			img_p[i] = Mother.Dice.nextInt(1000);
			img_p[i] = img_p[i] > 999 ? 0 : 1;
		}

		return new NdArray(img_p, new int[] { N, img_h, img_w }, false);
	}

	// １つの球状の模様を作成（ガウスですが）
	static double[] MakeOneCore(int i_ksize)
	{
		int max_xy = i_ksize;
		double sig = 5.0;
		double sig2 = sig * sig;
		double c_xy = 7;
		double[] core = new double[max_xy * max_xy];

		for (int px = 0; px < max_xy; px++) {
			for (int py = 0; py < max_xy; py++) {
				double r2 = (px - c_xy) * (px - c_xy) + (py - c_xy)	* (py - c_xy);
				core[py * max_xy + px] = Math.exp(-r2 / sig2) * 1;
			}
		}
		return core;
	}
}
