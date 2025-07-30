# Cross-asset-neural-model-
# A Cross-Asset Neural Model for Volatility and Risk Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Abstract

This project develops and evaluates a deep learning framework to model and forecast the key components of asset price dynamics—volatility and jumps—across multiple asset classes. Grounded in the theory of jump-diffusion processes, we use high-frequency intraday data for BTC, EUR/USD, and the S&P 500 to engineer features like Realized Volatility and Bipower Variation. A multi-stage modeling process reveals that standard LSTMs, even with Attention mechanisms, struggle with point forecasting in the face of major market regime shifts. Recognizing this, the project pivots to probabilistic forecasting, where the model predicts the parameters $(\mu, \sigma)$ of a distribution for future volatility. The model is trained using a Negative Log-Likelihood loss function. A formal Value-at-Risk (VaR) backtest on the final model reveals it systematically overestimates risk, a direct and measurable consequence of being trained through a period of extreme market stress. This project establishes a robust pipeline and provides a nuanced analysis of applying deep learning to risk management.

---

## 1. Theoretical Framework

The core motivation for this project is to create a data-driven model that captures the dynamics described by **jump-diffusion processes**. Unlike the geometric Brownian motion of Black-Scholes, a jump-diffusion model explicitly accounts for sudden, discontinuous price movements. The log-price, $p_t = \log(P_t)$, of an asset under such a model follows a stochastic differential equation (SDE) of the form:

$$
dp_t = \left(\mu - \frac{1}{2}\sigma^2\right)dt + \sigma dW_t + dJ_t
$$

Where:
*   $\mu$ is the drift coefficient.
*   $\sigma$ is the diffusion (volatility) coefficient.
*   $dW_t$ is the increment of a standard Wiener process, representing continuous market noise.
*   $dJ_t$ is a jump process, often a compound Poisson process, representing rare, large events.

Our goal is to use a neural network to forecast the future parameters of the diffusion ($\sigma$) and jump ($J_t$) components based on historical data.

---

## 2. Data and Feature Engineering

The project utilizes the **Intraday stock price data (minute bar)** dataset from [Kaggle](https://www.kaggle.com/datasets/arashnic/stock-data-intraday-minute-bar), focusing on BTC/USD, EUR/USD, and the S&P 500.

### 2.1 Data Pipeline

A robust pipeline was engineered to handle heterogeneous data sources, involving multi-format parsing, daily resampling, and timezone standardization.

### 2.2 Feature Engineering: From Theory to Data

We engineered daily features to create empirical proxies for the theoretical components of the jump-diffusion model.

#### **Idiosyncratic Features (Volatility & Jumps)**
For each asset, we calculated:
1.  **Daily Realized Volatility (RV):** A consistent estimator for the total quadratic variation of the price process. It is calculated as the sum of squared high-frequency log-returns $r_{t,i}$:
    $$
    RV_t = \sum_{i=1}^{N} r_{t,i}^2
    $$
2.  **Daily Jump Count:** To isolate jumps, we used a simple thresholding method. A return is flagged as a jump if it exceeds 4 times the local standard deviation of returns.

The plot below shows these engineered features for BTC, validating our approach by showing that high volatility periods coincide with more frequent jumps.

<img width="1484" height="983" alt="btcusd" src="https://github.com/user-attachments/assets/13c9b77f-a9b1-41a6-97b2-a86c862fabe4" />

*Figure 1: Daily realized volatility and detected intraday jumps for BTC/USD.*

#### **Cross-Asset Features (Contagion)**
To model market contagion, we engineered features like the 30-day rolling correlation between BTC and the S&P 500. The plot shows this relationship is highly dynamic, motivating the use of a learning-based model.

<img width="1275" height="558" alt="correlation btc and sp500" src="https://github.com/user-attachments/assets/c5a484fb-df5b-48dc-8dd6-a7bada18a2b6" />

*Figure 2: The 30-day rolling correlation between BTC and SPX returns is highly non-stationary.*

---

## 3. Modeling: From Point Forecasts to Probabilistic Risk Management

### 3.1 The Challenge: Regime Shifts and Model Limitations

Initial attempts to train standard and Attention-based LSTMs for point forecasting failed to generalize. This was due to a **major market regime shift** between the tranquil training period (pre-2017) and the volatile validation period (the 2017 crypto bull run). This led us to pivot from point forecasting to a more robust financial task.

### 3.2 The Final Model: A Probabilistic LSTM

Recognizing the difficulty of point forecasting, we developed a **Probabilistic LSTM**. Instead of predicting a single value, the model predicts the parameters—**mean $\mu$ and standard deviation $\sigma$**—of a Normal distribution for the next day's log-volatility:
$$
\log(1+RV_{t+1}) \sim \mathcal{N}(\mu_t, \sigma_t)
$$
where $(\mu_t, \sigma_t)$ are the outputs of the LSTM at time $t$.

The model is trained by minimizing the **Negative Log-Likelihood (NLL)** of the true data under the predicted distribution. For a Normal distribution, the NLL loss is:
$$
\mathcal{L}_{NLL} = \frac{1}{2} \sum_{i=1}^{N} \left( \log(2\pi\sigma_i^2) + \frac{(y_i - \mu_i)^2}{\sigma_i^2} \right)
$$
This trains the model to produce a distribution that makes the observed outcomes most probable.

<img width="1294" height="1286" alt="prob forecasts" src="https://github.com/user-attachments/assets/a7554028-82bb-442f-8998-71ca49637117" />
*Figure 3: The model's predicted mean (red, dashed) and 95% confidence interval (red, shaded) against the actual log-volatility (blue).*

---

## 4. Final Results: Value-at-Risk (VaR) Backtest

The ultimate test of a risk model is a formal backtest. We used the predicted distributions to calculate a 1-day 95% Value-at-Risk (VaR). The VaR is the 95th percentile of the predicted distribution:
$$
VaR_{95\%} = \mu_t + 1.645 \cdot \sigma_t
$$
A "breach" occurs if the actual volatility on the next day exceeds this VaR estimate. An accurate model should have a breach rate of 5%.

| Asset    | Total Days | Expected Breaches (5%) | Actual Breaches | Breach Rate | Result                   |
| :------- | :--------: | :--------------------: | :-------------: | :---------: | :----------------------- |
| **BTC**  |    295     |          14.8          |        7        |    2.37%    | **Overestimates Risk**   |
| **EURUSD**|    295     |          14.8          |        2        |    0.68%    | **Overestimates Risk**   |
| **SPX**  |    295     |          14.8          |        1        |    0.34%    | **Overestimates Risk**   |

### Interpretation of Results

The backtest provides the project's key insight. Our model is **systematically overestimating risk**. Having been "scarred" by the extreme volatility of the 2017 regime shift during training, it learned to be overly cautious. It consistently predicts a high level of uncertainty (a large `sigma`), pushing the VaR limit higher than necessary. This is a direct, measurable, and financially meaningful consequence of training a model through a market crisis.

---

## 5. Conclusion

This project successfully demonstrates an end-to-end quantitative research process. We conclude that:
1.  **Point forecasting of volatility is extremely difficult**, and increasing model complexity is not a panacea, especially across regime shifts.
2.  **Pivoting to probabilistic forecasting** provides a more robust framework for evaluating a model's understanding of risk.
3.  Our final deep learning model, when evaluated with an industry-standard VaR backtest, is found to be **overly conservative**, a direct result of the extreme events present in its training and validation history.

## 6. Future Work

*   **Model Calibration:** Since the model overestimates sigma, future work could focus on calibrating this output, for example, by using a different probability distribution (like the Student's t-distribution) that better fits the fat tails of financial returns.
*   **Explainability (SHAP):** Use a technique like SHAP to analyze the probabilistic model. Does the model increase its predicted uncertainty (`sigma`) in response to higher correlation or spillover features?
*   **Advanced Architectures:** Explore Transformer or Graph Neural Network (GNN) models to see if they can better capture the complex network effects of market contagion.

## 7. How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Set up Kaggle API Key:**
    *   Download your `kaggle.json` API token from your Kaggle account.
    *   Place the `kaggle.json` file in the root directory of this project.
3.  **Execute the Notebooks:**
    *   The project is structured as a series of Jupyter/Gradient notebooks.
    *   Run the cells in sequential order to replicate the full analysis.
