# VN30 Model Evaluation Report

## Scope

This report compares model outputs using out-of-sample walk-forward prediction rows currently available in `data/processed/`.

Important caution: the current dataset uses the current VN30 constituent list applied backward through time. The numbers below are real-data walk-forward results, but they are not final market evidence until survivorship bias, liquidity limits, and transaction costs are handled more carefully.

## Regression model comparison

Regression models are compared using Rank IC, top-5 hit rate, top-minus-bottom spread, turnover, transaction cost, after-cost return, Sharpe ratio, and max drawdown.

```text
       model_name  evaluated_dates  average_rank_ic  average_hit_rate  average_top_minus_bottom_spread  average_gross_top_n_return  average_after_cost_return  sharpe_ratio  max_drawdown  average_turnover  average_transaction_cost
gradient_boosting             1609         0.338033          0.676072                         0.043149                    0.023634                   0.022707     14.691060     -0.182220          0.927114                  0.000927
    random_forest             1609         0.312903          0.663642                         0.040575                    0.022402                   0.021573     13.398667     -0.158840          0.830348                  0.000830
      elastic_net             1609         0.236341          0.607209                         0.031172                    0.016626                   0.015711     10.428901     -0.189581          0.915423                  0.000915
            ridge             1609         0.227342          0.601989                         0.029869                    0.016208                   0.015241     10.287293     -0.172408          0.967662                  0.000967
```

## Classification model comparison

The logistic regression model is evaluated separately because it predicts top-quintile probability, not raw return.

```text
         model_name  evaluated_dates  average_precision  average_recall  average_selected_return  sharpe_ratio  max_drawdown
logistic_regression             1609           0.336234        0.280195                 0.013473       9.40319     -0.135127
```

## Baseline strategy comparison

The baseline strategy is evaluated as a portfolio return series, not as a ranking model.

```text
            strategy  evaluated_dates  average_portfolio_return  average_active_return  sharpe_ratio  max_drawdown  average_selected_count
       top5_momentum             1603                  0.007818               0.003623      3.092889     -0.873921                5.000000
    equal_weight_all             1613                  0.004414              -0.000000      2.126402     -0.904494               28.960322
       top5_reversal             1594                  0.004323              -0.000131      1.530188     -0.948201                5.000000
low_volatility_top10             1593                  0.003499              -0.000972      1.941379     -0.835925               10.000000
```

## Tree model feature importance

Feature importance shows which variables the tree models used most often for splits. It is useful for sanity checking, but it is not causal proof.

```text
       model_name                  feature  average_importance  importance_observations
gradient_boosting               return_60d            0.108755                     1609
gradient_boosting          rolling_vol_20d            0.102809                     1609
gradient_boosting                 drawdown            0.098642                     1609
gradient_boosting average_daily_volume_20d            0.091824                     1609
gradient_boosting  average_daily_value_20d            0.057248                     1609
gradient_boosting  estimated_ceiling_price            0.047842                     1609
gradient_boosting               return_10d            0.047787                     1609
gradient_boosting    estimated_floor_price            0.046471                     1609
gradient_boosting    distance_from_20d_low            0.046378                     1609
gradient_boosting          reference_price            0.043546                     1609
gradient_boosting   distance_from_20d_high            0.037464                     1609
gradient_boosting                return_3d            0.036320                     1609
gradient_boosting    traded_value_rank_20d            0.032236                     1609
gradient_boosting       rolling_return_20d            0.029177                     1609
gradient_boosting               return_20d            0.028078                     1609
gradient_boosting         volume_change_5d            0.025140                     1609
gradient_boosting        rolling_return_5d            0.021887                     1609
gradient_boosting                return_5d            0.020329                     1609
gradient_boosting              volume_z_20            0.017172                     1609
gradient_boosting        value_traded_z_20            0.015711                     1609
    random_forest               return_60d            0.111060                     1609
    random_forest          rolling_vol_20d            0.099392                     1609
    random_forest                 drawdown            0.095046                     1609
    random_forest average_daily_volume_20d            0.084729                     1609
    random_forest               return_10d            0.051018                     1609
    random_forest  estimated_ceiling_price            0.047537                     1609
    random_forest    distance_from_20d_low            0.046220                     1609
    random_forest  average_daily_value_20d            0.046136                     1609
    random_forest    estimated_floor_price            0.044944                     1609
    random_forest          reference_price            0.044606                     1609
    random_forest   distance_from_20d_high            0.038461                     1609
    random_forest                return_3d            0.037299                     1609
    random_forest    traded_value_rank_20d            0.032722                     1609
    random_forest               return_20d            0.030519                     1609
    random_forest       rolling_return_20d            0.030506                     1609
    random_forest         volume_change_5d            0.027661                     1609
    random_forest        rolling_return_5d            0.024637                     1609
    random_forest                return_5d            0.022785                     1609
    random_forest              volume_z_20            0.018214                     1609
    random_forest        value_traded_z_20            0.017590                     1609
```

## Provisional model choice

Best provisional regression model: gradient_boosting with average Rank IC 0.338033 and average after-cost top-5 return 0.022707.

The final model should be chosen using out-of-sample ranking quality and after-cost portfolio behavior, not in-sample fit. With real data, a model with slightly lower raw return but lower turnover may be preferable after transaction costs.
