import numpy as np

def generate_test_covariance_matrix(n_assets: int) -> np.ndarray:
    """
    Generate a valid test covariance matrix.
    
    Args:
        n_assets: Number of assets
        
    Returns:
        Positive definite covariance matrix
    """
    # Generar matriz de correlación base
    rng = np.random.RandomState(42)
    A = rng.normal(0, 1, (n_assets, n_assets))
    correlation = A @ A.T
    
    # Normalizar para obtener correlaciones válidas
    D = np.diag(1 / np.sqrt(np.diagonal(correlation)))
    correlation = D @ correlation @ D
    
    # Generar varianzas
    variances = rng.uniform(0.01, 0.04, n_assets)
    
    # Construir matriz de covarianza
    std_devs = np.sqrt(variances)
    covariance = correlation * np.outer(std_devs, std_devs)
    
    return covariance