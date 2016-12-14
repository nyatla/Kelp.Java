package jp.nyatla.kelpjava.common;

public class XorShift31Rand {
	private int seed;

	public XorShift31Rand(int i_seed) {
		this.seed = i_seed;
		for(int i=0;i<100;i++){
			this.nextInt();
		}
	}
	/**
	 * @return
	 * 0 to 0x7fffffff
	 */
	public int nextInt()
	{
		int y = this.seed;
		y = y ^ (y << 13);
		y = y ^ (y >> 17);
		y = y ^ (y << 5);
		this.seed = y;
		return y & 0x7fffffff;
	}
	public int nextInt(int i_max)
	{
		return this.nextInt()% i_max;
	}
	/**
	 * 
	 * @return
	 * 0/0x7fffffff to 1/0x7fffffff
	 */
	public double nextDouble()
	{
		return this.nextInt()*1.0/Integer.MAX_VALUE;
	}
}
