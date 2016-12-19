﻿package jp.nyatla.kelpjava.test;

    //ChainerのRNNサンプルを再現
    //https://github.com/pfnet/chainer/tree/master/examples/ptb
    class Test10
    {
        const int N_EPOCH = 39;
        const int N_UNITS = 650;
        const int BATCH_SIZE = 20;
        const int BPROP_LEN = 35;
        const int GRAD_CLIP = 5;

        public static void Run()
        {
            Console.WriteLine("Build Vocabulary.");

            Vocabulary vocabulary = new Vocabulary();

            int[] trainData = vocabulary.LoadData("data/ptb.train.txt");
            int[] validData = vocabulary.LoadData("data/ptb.valid.txt");
            int[] testData = vocabulary.LoadData("data/ptb.test.txt");

            int nVocab = vocabulary.Length;

            Console.WriteLine("Network Initilizing.");
            FunctionStack model = new FunctionStack(
                new EmbedID(nVocab, N_UNITS, name: "l1 EmbedID"),
                new Dropout(),
                new LSTM(N_UNITS, N_UNITS, name: "l2 LSTM"),
                new Dropout(),
                new LSTM(N_UNITS, N_UNITS, name: "l3 LSTM"),
                new Dropout(),
                new Linear(N_UNITS, nVocab, name: "l4 Linear")
            );

            //与えられたthresholdで頭打ちではなく、全パラメータのL2Normからレートを取り補正を行う
            GradientClipping gradientClipping = new GradientClipping(threshold: GRAD_CLIP);
            SGD sgd = new SGD(learningRate: 1.0);
            model.SetOptimizer(gradientClipping, sgd);

            double wholeLen = trainData.Length;
            int jump = (int)Math.Floor(wholeLen / BATCH_SIZE);
            int epoch = 0;

            Stack<NdArray[]> backNdArrays = new Stack<NdArray[]>();

            Console.WriteLine("Train Start.");

            for (int i = 0; i < jump * N_EPOCH; i++)
            {
                NdArray[] x = new NdArray[BATCH_SIZE];
                NdArray[] t = new NdArray[BATCH_SIZE];

                for (int j = 0; j < BATCH_SIZE; j++)
                {
                    x[j] = NdArray.FromArray(new[] { trainData[(int)((jump * j + i) % wholeLen)] });
                    t[j] = NdArray.FromArray(new[] { trainData[(int)((jump * j + i + 1) % wholeLen)] });
                }

                double sumLoss;
                backNdArrays.Push(new SoftmaxCrossEntropy().Evaluate(model.Forward(x), t, out sumLoss));
                Console.WriteLine("[{0}/{1}] Loss: {2}", i + 1, jump, sumLoss);

                //Run truncated BPTT
                if ((i + 1) % BPROP_LEN == 0)
                {
                    for (int j = 0; backNdArrays.Count > 0; j++)
                    {
                        Console.WriteLine("backward" + backNdArrays.Count);
                        model.Backward(backNdArrays.Pop());
                    }

                    model.Update();
                    model.ResetState();
                }

                if ((i + 1) % jump == 0)
                {
                    epoch++;
                    Console.WriteLine("evaluate");
                    Console.WriteLine("validation perplexity: {0}", Evaluate(model, validData));

                    if (epoch >= 6)
                    {
                        sgd.LearningRate /= 1.2;
                        Console.WriteLine("learning rate =" + sgd.LearningRate);
                    }
                }
            }

            Console.WriteLine("test start");
            double testPerp = Evaluate(model, testData);
            Console.WriteLine("test perplexity:" + testPerp);
        }

        static double Evaluate(FunctionStack model, int[] dataset)
        {
            FunctionStack predictModel = model.Clone();
            predictModel.ResetState();

            List<double> totalLoss = new List<double>();

            for (int i = 0; i < dataset.Length - 1; i++)
            {
                NdArray[] x = new NdArray[BATCH_SIZE];
                NdArray[] t = new NdArray[BATCH_SIZE];

                for (int j = 0; j < BATCH_SIZE; j++)
                {
                    x[j] = NdArray.FromArray(new[] { dataset[j + i] });
                    t[j] = NdArray.FromArray(new[] { dataset[j + i + 1] });
                }

                double sumLoss;
                new SoftmaxCrossEntropy().Evaluate(predictModel.Forward(x), t, out sumLoss);
                totalLoss.Add(sumLoss);
            }

            //calc perplexity
            return Math.Exp(totalLoss.Sum() / (totalLoss.Count - 1));
        }
    }
}
