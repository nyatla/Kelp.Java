package jp.nyatla.kelpjava.test;

import jp.nyatla.kelpjava.FunctionStack;
import jp.nyatla.kelpjava.functions.activations.Sigmoid;
import jp.nyatla.kelpjava.functions.connections.Linear;
import jp.nyatla.kelpjava.optimizers.MomentumSGD;


    //MLPによるXORの学習
    class Test1
    {
        public static void Run()
        {
            //訓練回数
            final int learningCount = 10000;

            //訓練データ
            double[][] trainData =
            {
                { 0.0, 0.0 },
                { 1.0, 0.0 },
                { 0.0, 1.0 },
                { 1.0, 1.0 }
            };

            //訓練データラベル
            double[][] trainLabel =
            {
                { 0.0 },
                { 1.0 },
                { 1.0 },
                { 0.0 }
            };

            //ネットワークの構成は FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                new Linear(2, 2,"l1 Linear"),
                new Sigmoid("l1 Sigmoid"),
                new Linear(2, 2,"l2 Linear")
            );

            //optimizerを宣言
            nn.SetOptimizer(new MomentumSGD());

            //訓練ループ
            System.out.println("Training...");
            for (int i = 0; i < learningCount; i++)
            {
                for (int j = 0; j < trainData.length; j++)
                {
                    //訓練実行時にロス関数を記述
                    Trainer.Train(nn, trainData[j], trainLabel[j], LossFunctions.SoftmaxCrossEntropy);
                }
            }

            //訓練結果を表示
            Console.WriteLine("Test Start...");
            foreach (double[] input in trainData)
            {
                NdArray result = Trainer.Predict(nn, input);
                int resultIndex = Array.IndexOf(result.Data, result.Data.Max());
                Console.WriteLine(input[0] + " xor " + input[1] + " = " + resultIndex + " " + result);
            }

            //学習の終わったネットワークを保存
            nn.Save("test.nn");

            //学習の終わったネットワークを読み込み
            FunctionStack testnn = FunctionStack.Load("test.nn");

            Console.WriteLine("Test Start...");
            foreach (double[] input in trainData)
            {
                NdArray result = Trainer.Predict(testnn, input);
                int resultIndex = Array.IndexOf(result.Data, result.Data.Max());
                Console.WriteLine(input[0] + " xor " + input[1] + " = " + resultIndex + " " + result);
            }

        }
    }
}
