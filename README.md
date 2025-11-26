# Asynchronous-Unsourced-Multiple-Access-Latency-Constraints-and-Delay-Information

This repository contains the code for the numerical evaluations of the papers:

[1] J.-S. Wu, P.-H. Lin, M. A. Mross, and E. A. Jorswieck, "Asynchronous unsourced multiple access: latency-constraints and delay information," *submitted to IEEE Trans. Inf. Theory*, Nov. 2025.

This paper was presented in part in

[2] J.-S. Wu, P.-H. Lin, M. A. Mross, and E. A. Jorswieck, “Worst-case per-user error bound for asynchronous unsourced multiple access,” in
Proc. IEEE Int. Symp. Inf. Theory (ISIT), Jul., 2024, pp. 3207–3212.

and

[3] J.-S. Wu, P.-H. Lin, M. A. Mross, and E. A. Jorswieck, "Wrap-decoding in asynchronous unsourced multiple access with and without delay information,” in Proc. IEEE Int. Symp. Inf. Theory (ISIT), Jun., 2025

## Content of the repository

This repository provides code to evaluate random-coding bounds on the per-user probability of error (PUPE) of asynchronous unsourced multiple-access channels (AUMACs). 

1. AUMAC_with_latency_constraint.py provides the upper bound on the PUPE for Theorem 2 in [1].
2. AUMAC_Wrap_Decoding.py provides the upper bound on the PUPE for Theorems 5 and 6 in [1].
3. Corollary.py provides the upper bound on the PUPE for Corollary in [1].
4. AUMAC_Converse.m provides the converse bounds in [1].
5. /SDA/: cyclic\_decoding\_1.m computes the upper bound on the PUPE in [3] corresponding to the first term in Remark 1 of [1] and [3]. cyclic\_decoding\_2.m then computes the full upper bound in [3], using the first-term contribution obtained from cyclic\_decoding\_1.m.
6. Polyanskiy.py provides the upper bound on the PUPE in [4] Y. Polyanskiy, "A perspective on massive random-access," in Proc. IEEE Int. Symp. Inf. Theory (ISIT), 2017, Aachen, Germany, pp. 2523-2527.
7. CCS_AMP_BP_AUMAC.py provides the upper bound on the PUPE by the CCS-AMP-BP scheme, which was originally proposed in [6]. We extend this scheme to the AUMAC system.
  [6]. V. K. Amalladinne, A. K. Pradhan, C. Rush, J.-F. Chamberland, and K. R. Narayanan, “Unsourced random access with coded compressed
       sensing: Integrating AMP and belief propagation,” IEEE Trans. Inf. Theory, vol. 68, no. 4, pp. 2384–2409, 2022.

**Note**: The Eb/N0 of the CCS-AMP-BP scheme for the UMAC is copied from [6], and hence not included in this repository. 

**Version**: The simulations are performed with Python 3.13.4 and pytorch 2.7.1.
