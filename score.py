import logging
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def w(x):
    if "_SOUND" in x:
        return 8.94 # weighting  Healthy grains more important than defective grains
    return 1

def scoring_fn(df):
    """
    Weighted Accuracy Metric 90% Healthy grains, 10% unhealthy grains
    """
    df['weight'] = df.cls.apply(w)
    return accuracy_score(df.cls,df.prediction,sample_weight=df.weight)

if __name__ == "__main__":
    """Scoring Function
    This function is called by Unearthed's SageMaker pipeline. It must be left intact.
    """
    parser = argparse.ArgumentParser()
    # Avoid pipeline errors which assumes all scoring functions have an actual parameter.
    parser.add_argument("--actual", type=str, default="not-used")
    parser.add_argument(
        "--predicted",
        type=str,
        default="/opt/ml/processing/input/predictions/public.csv.out",
    )
    parser.add_argument(
        "--output", type=str, default="/opt/ml/processing/output/scores/public.txt"
    )
    args = parser.parse_args()

    df_pred = pd.read_csv(args.predicted , header=None, index_col=False)
    df_pred.columns = ['file_name', 'path', 'cls', 'prediction', 'proba_1', 'prediction2',
       'proba_2', 'prediction3', 'proba_3']
    logger.info(f"predictions have shape of {df_pred.shape}")

    score = scoring_fn(df_pred)
    print ("Accuracy Score:" + str(score))
    # write to the output location
    with open(args.output, "w") as f:
        f.write(str(score))

