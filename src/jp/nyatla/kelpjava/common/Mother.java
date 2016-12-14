package jp.nyatla.kelpjava.common;




/**
 * 乱数の素 C#ではRandomを複数同時にインスタンスすると似たような値しか吐かないため 一箇所でまとめて管理しておく必要がある
 */
public class Mother {
//	public static Random Dice = new Random(128);
	public static XorShift31Rand Dice = new XorShift31Rand(128);
	static double Alpha, Beta, BoxMuller1, BoxMuller2;
	static boolean Flip;
	public static double Mu = 0.0;
	public static double Sigma = 1.0;

	/**
	 * 平均mu, 標準偏差sigmaの正規分布乱数を得る。Box-Muller法による。
	 */
	public static double RandomNormal()
	{
		if (!Flip) {
			Alpha = Dice.nextDouble();
			Beta = Dice.nextDouble() * Math.PI * 2;
			BoxMuller1 = Math.sqrt(-2 * Math.log(Alpha));
			BoxMuller2 = Math.sin(Beta);
		} else {
			BoxMuller2 = Math.cos(Beta);
		}

		Flip = !Flip;

		return Sigma * (BoxMuller1 * BoxMuller2) + Mu;
	}
}
