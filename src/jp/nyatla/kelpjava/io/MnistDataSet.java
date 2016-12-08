package jp.nyatla.kelpjava.io;

public class MnistDataSet {
    public Array[] Data;
    public int[][] Label;

    public MnistDataSet(Array[] data, int[][] label)
    {
        this.Data = data;
        this.Label = label;
    }
}
