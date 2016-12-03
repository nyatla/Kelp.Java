package jp.nyatla.kelpjava;

/**
 * Optimizerの基底クラス
 * 
 */
public interface IOptimizer
{
	void update(OptimizeParameter i_parameter);
	IOptimizer initialise(OptimizeParameter i_parameter);
}
