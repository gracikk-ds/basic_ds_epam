from IPython.display import display, Math, Latex
import numpy as np
import pandas as pd

def estimate_parameters_MLE(sample):
    x1 = str(np.round(sample[0], 3))
    xn = str(np.round(sample[-1], 3))
    print('In general case for Normal distribution: ')
    display(Math(r'\hat{\theta}(X_{[n]})=\arg\max_{\theta \in \Theta} \ln L\; (x_1,\ldots, x_n;\; \theta) ='))
    display(Math(r'\arg\max_{\theta \in ((-\infty, \infty), [0, \infty))} \Big[\ln {\frac {1}{\theta_2 {\sqrt {2\pi }}}}\;e^{-{\tfrac {(x_1-\theta_1 )^{2}}{2\theta_2 ^{2}}}} + \dots + ' + \
                 r'\ln {\frac {1}{\theta_2 {\sqrt {2\pi }}}}\;e^{-{\tfrac {(x_n-\theta_1 )^{2}}{2\theta_2 ^{2}}}} \Big] ='))
    display(Math(r'\arg\max_{\theta \in ((-\infty, \infty), [0, \infty))} -\frac{n}{2} \ln 2\pi - n \ln\theta_2 - \frac{1}{2\theta_2^2} \cdot \sum_{i=1}^{n} (x_i - \theta_1)^2 = '))
    display(Math(r'\arg\min_{\theta \in ((-\infty, \infty), [0, \infty))}  n \ln\theta_2 + \frac{1}{2\theta_2^2} \cdot \sum_{i=1}^{n} (x_i - \theta_1)^2 '))
    
    print('\n\n For our sample: ')
    display(Math(r'\hat{\theta}(X_{[n]})=\arg\max_{\theta \in \Theta} \ln L\; (' +  x1 + r',\ldots, ' + xn + r';\; \theta) ='))
    display(Math(r'\arg\min_{\theta \in ((-\infty, \infty), [0, \infty))} 100\ln\theta_2 + \frac{1}{2\theta_2^2} \cdot \Big[(' + x1 \
                 + r' - \theta_1)^2 + \dots (' + xn + r' - \theta_1)^2 \Big]'))
# display(Math(r'F(k) = \int_{-\infty}^{\infty} f(x) e^{2\pi i k} dx'))

    
def display_hist_formula():
    display(Math(r'\begin{equation*} f_n(x) =   \begin{cases}    0, \quad x < z_1, \\' + 
                r'\frac{n_1}{n(z_2 - z_1)}, \quad x \in [z_1, z_2),\\ ' + 
                r'\dots, \\' + 
                r'\frac{n_k}{n(z_{k+1} - z_k)}, \quad x \in [z_k, z_{k+1}),\\' + 
                r'\dots, \\' + 
                r'0, \quad x \geq z_{k+1}    \end{cases}    \end{equation*}'))
                
                
def display_abshist_formula():
    display(Math(r'\begin{equation*} f_n(x) =   \begin{cases}    0, \quad x < z_1, \\' + 
                r'n_1, \quad x \in [z_1, z_2),\\ ' + 
                r'\dots, \\' + 
                r'n_k, \quad x \in [z_k, z_{k+1}),\\' + 
                r'\dots, \\' + 
                r'0, \quad x \geq z_{k+1}    \end{cases}    \end{equation*}'))
                
                
def display_sample_f():
    display(Math(r'\begin{equation*} F_n(x) =   \begin{cases}    0, \quad x < x_{(1)}, \\' + 
                r'\frac{1}{n}, \quad x \in [x_{(1)}, x_{(2)}),\\ ' + 
                r'\dots, \\' + 
                r'\frac{k}{n}, \quad x \in [x_{(k)}, x_{(k+1)}),\\' + 
                r'\dots, \\' + 
                r'1, \quad x \geq x_{(n+1)}   \end{cases}    \end{equation*}'))
    
    
def get_sample_kstest():
    np.random.seed(42)
    return np.random.normal(5, 4, size=100)


def get_sample_mannwhitney():
    np.random.seed(42)
    sample_t = np.random.standard_t(df=50, size=100)
    
    np.random.seed(42)
    sample_n = np.random.normal(size=100)
    
    return sample_t, sample_n


def print_vars_types():
    print('*\tDependent / target features, those \033[1m the model is trained \033[0m for predicting.\n*\tFactors / independent variables / features, those \033[1m based on\033[0m which the model is trained.')
    
def print_vars_types2():
    print('*\tNumerical continious features\n*\tNumerical discrete features\n*\tCategorical nominal features\n*\tCategorical ordinal features')