# Note on A NICE-MC

## Introduction

There are three methods to generate complex probability distributions in machine learning, they're variational inference (VI), Monte Carlo (MC) methods and Generative Adversaial Nets (GAN). In VI, complex probability distributions are approximated using tractable probability distributions, example is variational autoencoders (VAE) [^1].  MC methods use step by step ancestrally sampling from a proposal distribution. While GAN are a Markov-Chain-free process to generate new samples using two networks (critic network and generatative network)  competing each other[^2] [^3]. 

While new advance is made in both VI and GAN in rencent years, the MC method hasn't been much boosted by deep neuron network.NICE-MC was proposed to accelerate MC. It's a GAN-like method to optimize the generator network with a different loss function. And to tackle the detailed balance problem, they use a generator network architecture called non-linear independent components estimation(NICE), which can be easily inversd and has a easy-to-calculate Jacobian determinant[4].

## Jacobian Matrix And Volume Preserve

The Jacbian matrix is introducted to explain why NICE network can preserve detailed balance. Before NICE network is detailed introducted, we can view Hamiltonian Monte Carlo (HMC) medthod as an example. In HMC method, an auxiliary variable $v$ was sampled from a factored Gaussian distribution. Then $(x', v')$ can be obtained by simulating the dynamics corresponding to the Hamiltonian

$$
H(x,v) = v^{T}v/2 + U(x)
$$
where $x$ and $v$ are iteratively updated using the _leapfrog_ intergrator. The transition from $(x,v)$ to $(x',v')$ is deterministic, invertible and volume preserving, which means that 
$$
g_{\theta}(x',v'|x,v) = g_{\theta}(x,v|x',v') 
$$
_Proof_ :  Jacobian is derivative defined on functions $f: R^{m} \rightarrow R^{n}$, has the form like
$$
J = \delta f =\left[
\begin{matrix}
\frac{\partial{f_{1}}}{\partial{x_{1}}} & \frac{\partial{f_{1}}}{\partial{x_{2}}} & \cdots & \frac{\partial{f_{1}}}{\partial{x_{m}}} \\
\frac{\partial{f_{2}}}{\partial{x_{1}}} & \frac{\partial{f_{2}}}{\partial{x_{2}}} & \cdots & \frac{\partial{f_{2}}}{\partial{x_{m}}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial{f_{n}}}{\partial{x_{1}}} & \frac{\partial{f_{n}}}{\partial{x_{2}}} & \cdots & \frac{\partial{f_{n}}}{\partial{x_{m}}}
\end{matrix}
\right]
$$
The function $f$ stands for a transformation of a $R^{m}$ space into a $R^{n}$ space, and the unit volume ratio between the two space is $det(J)$. If the function $(x,y) = f(u,v) $ is defined on $R^{2} \rightarrow R^{2}$, then 
$$
J(u,v) = 
\left[
\begin{matrix}
x_{u} & x_{v} \\
y_{u} & y_{v}
\end{matrix}
\right]
$$
then
$$
\begin{equation}
\det(J) = \frac{\partial{x}}{\partial{u}}\frac{\partial{y}}{\partial{v}} - \frac{\partial{x}}{\partial{v}}\frac{\partial{y}}{\partial{u}}
\end{equation}
$$
the unit vector
$$
\begin{equation}
du = (du,0) \\
dv = (dv,0)
\end{equation}
$$
has been transformed into 
$$
\begin{equation}
dx = J(u,v)du \\
dy = J(u,v)dv
\end{equation}
$$
the unit area after transformation is
$$
\begin{equation}
dA = ||dx\times dy|| = |x_{u}y_{v} - x_{v}y_{u}|dudv
\end{equation}
$$
![test](./test.png)

 So if $h$$p_{H}(h)$ is the probability distributions and $\textbf{x}=f^{-1}(\textbf{h})$, then the probability distributions of $x$ is
$$
p_{X}(\textbf{x})=p_{H}(f(\textbf{x}))\left|\det\left(\frac{\partial(f(\textbf{x}))}{\partial{\textbf{x}}}\right)\right|
$$
In our example, we solve the Hamiltonian to get $(x',v')​$ from $(x,v)​$, specifically
$$
\begin{equation}
\left\{
\begin{array}\\
x' = x + \dot xdt \\
v' = v + \dot vdt 
\end{array}
\right.
\end{equation}
$$
Then, $\textbf{x} = (x,v)$ and $\textbf{h} = (x',v')$. In the Hamiltonian dynamics, we have
$$
\begin{equation}
\dot x = \frac{\partial{H}}{\partial{v}} \\
\dot v = -\frac{\partial{H}}{\partial{x}}
\end{equation}
$$
So
$$
\begin{equation}
\left\{
\begin{array}\\
x' = x +\frac{\partial{H}}{\partial{v}}  dt \\
v' = v - \frac{\partial{H}}{\partial{x}}  dt 
\end{array}
\right.
\end{equation}
$$
To better understand NICE network in the later section and generalize the conclusion, here we rewrite it in the following way
$$
\begin{equation}
\left\{
\begin{array}\\
x' = x+m^{(1)}(x,v) \\
v' = v+m^{(2)}(x,v)
\end{array}
\right.
\end{equation}
$$


Then Jacobian is
$$
J = 
\left[
\begin{matrix}
\frac{\partial{x'}}{\partial{x}} & \frac{\partial{v'}}{\partial{x}} \\
\frac{\partial{x'}}{\partial{v}} & \frac{\partial{v'}}{\partial{v}}
\end{matrix}
\right]
=
\left[
\begin{matrix}
1+\frac{\partial{m^{(1)}}}{\partial{x}} & \frac{\partial{m^{(1)}}}{\partial{v}} \\
\frac{\partial{m^{(2)}}}{\partial{x}} & 1+\frac{\partial{m^{(2)}}}{\partial{v}}
\end{matrix}
\right]
$$
The condition $\det(J) = 1$ then implies 
$$
\begin{equation}
\left(1+\frac{\partial{m^{(1)}}}{\partial{x}}\right)\left(1+\frac{\partial{m^{(2)}}}{\partial{v}}\right) - \frac{\partial{m^{(1)}}\partial{m^{(2)}}}{\partial{x}\partial{v}} = 1

\end{equation}
$$
which leads to
$$
\begin{equation}
\frac{\partial{m^{(1)}}}{\partial{x}} + \frac{\partial{m^{(2)}}}{\partial{v}} = 0 \tag{1}
\end{equation}
$$
So, if $m^{(1)} = \frac{\partial{H}}{\partial{v}}$ and $m^{(2)} = -\frac{\partial{H}}{\partial{x}}$ , the equation above is satisfied, the $\det(J) = 1$. Then the $P((x,v)) = P((x',v'))$. At every $x$ the transition probability $P(x'|x)$ only depend on the $v$ sampled from Gaussian distribution. So the detailed balance is preserved. 

## NICE Network

Then, the proposal of NICE Network architecture seems obvious. Refer to the equation (1). To satisfy this equation, we can let
$$
\begin{equation}
\left\{
\begin{matrix}
\frac{\partial{m^{(1)}}}{\partial{x_{1}}} = 0 \\
\frac{\partial{m^{(2)}}}{\partial{x_{2}}} = 0
\end{matrix}
\right.
\end{equation}
$$
While NICE network has easy-to-calculate Jacobian determinant, it also can be trivially inversd. If we let 
$$
\begin{equation}
\left\{
\begin{array}\\
y_{1} = x_{1} \\
y_{2} = x_{2} + m(x_{1})
\end{array}
\right.
\end{equation}
$$
then the inverse of the transformation is
$$
\begin{equation}
\left\{
\begin{array}\\
x_{1} = y_{1} \\
x_{2} = y_{2} - m(y_{1})
\end{array}
\right.
\end{equation}
$$
So that 
$$
\begin{equation}
\begin{array}\\
m^{(1)} = 0 \\
m^{(2)} = m(x_{1})
\end{array}
\end{equation}
$$
Then it is easy to check this satisfy equation(1). And the $m(x)$ can be any function.

As part of the input are unchanged, we need to exchange the role of the two parts, so that the composition of this transformation can modifies every dimension.

In the NICE-MC, a NICE network architecture is used as following

![2017-08-18 22-00-40 的屏幕截图](./2017-08-18 22-00-40 的屏幕截图.png)

## WGAN-GP

In the original paper, a loss function for GAN was proposed as follow:
$$
\min_{G}\max_{D} L(D,G) = E_{x\sim P_{data(x)}}[\log D(x)] + E_{z\sim P_{z}(z)}[1-\log D(G(z))]
$$
And it's other form can be written as:
$$
\min_{G}\max_{D} L(D,G) = E_{x\sim P_{data(x)}}[\log D(x)] - E_{z\sim P_{z}(z)}[\log D(G(z))]
$$
Note that $D$ is a discriminator who gives a percentage at which the data provided are from the really data or not, while $G$ is a generator who generates fake data trying to fool the discriminator. 

While GAN proposed a different view of composing loss function in generative model, it suffers from problems such as slow training and model collapse which means $G$ will only propose a single data sample.

So, a series of following work trying to solve these problems, in which [^5][^6] gave a thorough analysis of the original GAN and proposed WGAN. In WGAN, the loss function has been changed to
$$
\min_{G}\max_{D} L(D,G) = E_{x\sim P_{data(x)}}[D(x)] - E_{z\sim P_{z}(z)}[D(G(z))]
$$
Notice that the $\log$ function has been removed. And the discriminator network is no long output a percentage but give a score of which higher means more likely for the data to be really. Then the weight of the D network should be cliped in a certain range, e.g. [-0.001,0.001], this is called weight clipping.

However, WGAN still sometimes suffer from problems such as slow training. The following work of WGAN changed weight clipping to gradient penalty [^7], which add a term in the loss function to constrain the D network. 
$$
||\bigtriangledown_x D(x)|| \le K
$$
So the finally loss function for D is (assume $K=1$)
$$
L(D) = - E_{x \sim P_{data}}[D(x)]+E_{x\sim P_{g}}[D(x)] + \lambda E_{x\sim \chi}[||\bigtriangledown_{x} D(x)||-1]^2
$$
And the loss function for G is
$$
L(G) = E_{x\sim P_g}[D(x)]
$$
In the loss function of D, the last term $E_{x\sim \chi}[||\bigtriangledown_{x} D(x)||-1]^2$ requires sampling samples from the whole $x$ space, which is not tractable. But the author of [7] propsed that it is not necessary to sample the whole space, but concentrate in samples that come from really data, fake data and data space between they. So we can first sample a random number $\epsilon$ from $[0,1]$, and sample a really date $x_{r}$ and a fake date $x_g$, then new sample can be wrote as
$$
\hat{x} = \epsilon x_{r} + (1-\epsilon)x_g
$$
So the loss function for D can be wrote as 
$$
L(D) = - E_{x \sim P_{data}}[D(x)]+E_{x\sim P_{g}}[D(x)] + \lambda E_{x\sim P_{\hat{x}}}[||\bigtriangledown_{x} D(x)||-1]^2
$$

## NICE-MC

Different from HMC method which sample a random auxiliary $v$ and use $v$ to update $x$, in NICE-MC because the network are trained so that it may prefer some $x$ over others even with random sampled $v$, to aviod this, we first sample a random variable $u\sim Uniform[0,1]$, then if $u\lt0.5$, $(x',v') = f_\theta(x,v)$, otherwise $(x',v')=f_\theta^{-1}(x,v)$, so that every proposal satisfy
$$
\begin{equation}
g_\theta(x',v'|x,v) = g_\theta(x,v|x',v')
\end{equation}
$$
Then we can use Metropolis-Hastings method to decide whether to accept this proposal.

![2017-08-19 22-57-16 的屏幕截图](./2017-08-19 22-57-16 的屏幕截图.png)

*Proof*: We know that $|\det(J)|=|\det(\frac{\partial{f(x,v)^{-1}}}{\partial{(x,v)}})|=1$. So
$$
\begin{align*}\\
g(x',v'|x,v)  &= \frac{1}{2}\left|\det\frac{\partial{f(x,v)^{-1}}}{\partial(x,v)}\right|\mathbb{I}(x',v'=f(x,v)) +\frac{1}{2}\left|\det\frac{\partial{f(x,v)}}{\partial(x,v)}\right|\mathbb{I}(x',v'=f^{-1}(x,v)) \\
&=\frac{1}{2}\mathbb{I}(x',v'=f(x,v))+\frac{1}{2}\mathbb{I}(x',v'=f^{-1}(x,v)) \\
&=\frac{1}{2}\mathbb{I}(x,v=f^{-1}(x',v')) + \frac{1}{2}\mathbb{I}(x,v=f(x',v')) \\
&=g(x,v|x',v')
\end{align*}
$$
The NICE-MC method use the NICE network architecture mentioned before, and use a WGAN-like loss function to optimize parameters of the NICE network. The loss function proposed is
$$
\begin{equation}
\min_{\theta}\max_{D}E_{x\sim p_d}[D(x)] - \lambda E_{\bar{x}\sim \pi^{b}_{\theta}}[D(\bar{x})]-(1-\lambda)E_{x_d\sim p_d,\bar{x}\sim T^{m}_\theta(\bar{x}|x_d)}[D(\bar{x})]
\end{equation}
$$
where $\lambda\in(0,1)$, $b\in N^{+}$, $m\in N^+$ are hyperparameters, $\bar{x}$ denotes "fake" samples from the generator.

There are two type of generated samples:

1. Samples obtained after $b$ transitions $\bar{x}\sim\pi^b_\theta$, starting from $x_0\sim\pi^0$;
2. Samples obtained after $m$ transitions, starting from a data sample $x_d\sim p_d$.

Also to constrain the auxiliary $v$, we minimize the distance between the distance for the generated $v'$ and the prior distribution $p(v)$ (which is a factored Gaussian):
$$
\min_{\theta}\max_{D}L(x|\theta,D) + \eta L_d(p(v),p_\theta(v'))
$$
where $L$ is the MGAN objective, $L_d$ is an objective that measures the divergence between two distributions and $\eta$ is a parameter to balance between the two factors (we use KL divergence for $L_d$).

Then we can write the loss function for $D (\lambda=1$)
$$
\begin{equation}
\min_\theta E_{x\sim p_{fake}}[D(x)]  - E_{x\sim{p_d}}[D(x)] + E_{x\sim{\hat{x}}}[||\bigtriangledown_xD(x)||-1]^2
\end{equation}
$$
The loss function for $G$ is
$$
\min_\theta E_{x\sim p_{fake}}[D(x)] +\eta L_d(p(v),p_\theta(v'))
$$
In NICE-MC, to aviod autocorrelation we give discriminator two samples $(x_1,x_2)$ at a time. This is similar to minibatch discrimination trick in GAN. For "real" date we draw two independent samples; for "fake" date we draw $x_2\sim T_\theta^m(x_2|x_1)$ such that $x_1$ is either drawn from the "real" data distribution or from samples after running the chain for $b$ steps, and $x_2$  is the sample after running the chain for $m$ steps, to be specifical, “fake” data contain two types of samples:

* Starting from a data point $x$, sample $z_1$ in B steps.
* Starting from some noise $z$, sample $z_2$ in B steps; and from $z_2$ sample $z_3$ in M steps.

![](屏幕快照 2017-08-20 18.45.32.png)

Then there is only one question left, where can we get the original "real" data? To get original data, we can first run untrained NICE-MC, note that without training NICE-MC can still yield right data, and use the data as "real" data. Then for certain iteration we train the NICE network, but drop the Metropolis-Hastings process, just optimize the loss function. Then with a better trained NICE network we then run NICE-MC for certain iteration and use the data produced as "real" data. Then train the NICE network again and repeat this process.

##References

[^1]: arxiv 1606.05908
[^2]: arxiv 1406.2661

[^3]: arxiv 1701.07875
[^ 4]: arxiv 1410.8516 
[^ 5]: arxiv 1701.04862
[^ 6]: arxiv 1701.07875
[^ 7]: arxiv 1704.00028



