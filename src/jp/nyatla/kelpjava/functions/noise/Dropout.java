package jp.nyatla.kelpjava.functions.noise;

import java.util.ArrayList;
import java.util.List;

import jp.nyatla.kelpjava.common.Mother;
import jp.nyatla.kelpjava.common.NdArray;
import jp.nyatla.kelpjava.functions.common.Function;
import jp.nyatla.kelpjava.functions.common.FunctionParameter;

/**
 * [Serializable]
 * 
 */
public class Dropout extends Function
{
	private static final long serialVersionUID = -7173439726306795015L;
	private final double dropoutRatio;
	private final List<double[]> maskStack = new ArrayList<double[]>();
	protected Dropout(Dropout i_src)
	{
		super(i_src);
		this.dropoutRatio=i_src.dropoutRatio;
		for(int i=0;i<i_src.maskStack.size();i++){
			this.maskStack.add(i_src.maskStack.get(i).clone());
		}
		return;
	}
	
	public Dropout() {
		this("Dropout");
	}
	public Dropout(String i_name) {
		this(0.5,i_name);
	}

	public Dropout(double i_dropoutRatio, String i_name) {
		super(i_name);
		this.parameters=new FunctionParameter[]{};
		this.dropoutRatio = i_dropoutRatio;
	}

	@Override
	protected NdArray[] forwardSingle(NdArray[] i_x)
	{
		NdArray[] result = new NdArray[i_x.length];
		double[] mask = new double[i_x[0].length()];
		double scale = 1.0 / (1.0 - this.dropoutRatio);

		for (int i = 0; i < mask.length; i++) {
			mask[i] = Mother.Dice.nextDouble() >= this.dropoutRatio ? scale : 0;
		}

		for (int i = 0; i < i_x.length; i++) {
			double[] y = new double[i_x[i].length()];

			for (int j = 0; j < mask.length; j++) {
				y[j] = i_x[i].data[j] * mask[j];
			}

			result[i] = new NdArray(y, i_x[i].shape.clone(), false);
		}

		this.maskStack.add(mask);

		return result;
	}

	@Override
	protected NdArray[] backwardSingle(NdArray[] gy) {
		NdArray[] result = new NdArray[gy.length];

		double[] mask = this.maskStack.get(this.maskStack.size() - 1);
		this.maskStack.remove(this.maskStack.size() - 1);

		for (int i = 0; i < gy.length; i++) {
			double[] gx = new double[gy[i].length()];

			for (int j = 0; j < mask.length; j++) {
				gx[j] = gy[i].data[j] * mask[j];
			}

			result[i] = new NdArray(gx, gy[i].shape.clone(),false);
		}

		return result;
	}

	@Override
	public NdArray[] predict(NdArray[] input) {
		// nothing to do
		return input;
	}

	@Override
	public Object deepCopy() {
		return new Dropout(this);
	}

}
