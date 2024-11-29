import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta
import sys

# Añadir el directorio raíz al path de Python
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_management import DataManager

@pytest.fixture
def sample_data():
    """Genera datos sintéticos para testing."""
    # Generar 2 años de datos diarios para asegurar suficientes datos mensuales
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    np.random.seed(42)
    data = {}
    for symbol in symbols:
        trend = np.linspace(100, 120, len(dates))
        volatility = np.random.normal(0, 2, len(dates))
        prices = trend + volatility
        data[symbol] = prices

    df = pd.DataFrame(data, index=dates)
    return df, symbols

@pytest.fixture
def test_data_dir(tmp_path):
    """Crea un directorio temporal para los archivos de test."""
    return tmp_path / "test_data"

@pytest.fixture
def mock_yf_download(monkeypatch, sample_data):
    """Mock para yfinance.download."""
    def mock_download(*args, **kwargs):
        df, _ = sample_data
        columns = pd.MultiIndex.from_product([['Adj Close'], df.columns])
        df_multi = pd.DataFrame(df.values, index=df.index, columns=columns)
        return df_multi
    
    monkeypatch.setattr(yf, "download", mock_download)

class TestDataManagerInit:
    def test_basic_initialization(self):
        """Test de inicialización básica de DataManager."""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        dm = DataManager(
            symbols=symbols,
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        assert dm.symbols == symbols
        assert dm.start_date == '2023-01-01'
        assert dm.end_date == '2023-12-31'
        assert dm.prices is None
        assert dm.returns is None
        assert dm.statistics is None

    def test_data_directory_creation(self, test_data_dir):
        """Test que el directorio de datos se crea correctamente."""
        dm = DataManager(
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_dir=test_data_dir
        )
        
        assert test_data_dir.exists()
        assert test_data_dir.is_dir()

class TestDataDownload:
    def test_successful_download(self, mock_yf_download):
        """Test de descarga exitosa de datos."""
        dm = DataManager(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        dm.download_data()
        assert dm.prices is not None
        assert not dm.prices.empty
        assert all(symbol in dm.prices.columns for symbol in dm.symbols)

    def test_missing_data_handling(self, monkeypatch, sample_data):
        """Test del manejo de datos faltantes."""
        df, symbols = sample_data
        # Introducir valores faltantes
        df.iloc[10:20, 0] = np.nan
        
        def mock_download(*args, **kwargs):
            columns = pd.MultiIndex.from_product([['Adj Close'], df.columns])
            df_multi = pd.DataFrame(df.values, index=df.index, columns=columns)
            return df_multi
        
        monkeypatch.setattr(yf, "download", mock_download)
        
        dm = DataManager(
            symbols=symbols,
            start_date='2022-01-01',  # Ajustado para coincidir con sample_data
            end_date='2023-12-31'
        )
        
        dm.download_data()
        assert not dm.prices.isnull().any().any()

class TestDataCalculations:
    def test_returns_calculation(self, mock_yf_download):
        """Test del cálculo de retornos."""
        dm = DataManager(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        dm.download_data()
        dm.calculate_returns()
        
        assert dm.returns is not None
        assert not dm.returns.empty
        assert isinstance(dm.returns, pd.DataFrame)

    def test_statistics_calculation(self, mock_yf_download):
        """Test del cálculo de estadísticas."""
        dm = DataManager(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        dm.download_data()
        dm.calculate_returns()
        stats = dm.calculate_statistics()
        
        assert stats is not None
        assert 'expected_returns' in stats
        assert 'covariance_matrix' in stats
        assert 'symbols' in stats
        assert stats['covariance_matrix'].shape == (3, 3)

class TestDataStorage:
    def test_save_and_load_data(self, test_data_dir, mock_yf_download):
        """Test de guardado y carga de datos."""
        dm = DataManager(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_dir=test_data_dir
        )
        
        # Procesar y guardar datos
        dm.process_all()
        
        # Crear nueva instancia y cargar datos
        dm_new = DataManager(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_dir=test_data_dir
        )
        
        assert dm_new.load_data()
        assert dm_new.prices is not None
        assert dm_new.returns is not None
        assert dm_new.statistics is not None
        
        # Verificar que los datos son iguales y mantener el tipo de índice
        pd.testing.assert_index_equal(dm.prices.index, dm_new.prices.index)
        pd.testing.assert_frame_equal(
            dm.prices.reset_index(drop=True), 
            dm_new.prices.reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            dm.returns.reset_index(drop=True), 
            dm_new.returns.reset_index(drop=True)
        )

class TestEndToEnd:
    def test_complete_process(self, test_data_dir, mock_yf_download):
        """Test del proceso completo de datos."""
        dm = DataManager(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_dir=test_data_dir
        )
        
        # Ejecutar proceso completo
        stats = dm.process_all()
        
        # Verificar resultados
        assert dm.prices is not None
        assert dm.returns is not None
        assert stats is not None
        assert stats['expected_returns'].shape == (3,)
        assert stats['covariance_matrix'].shape == (3, 3)
        assert len(stats['symbols']) == 3

    def test_data_consistency(self, test_data_dir, mock_yf_download):
        """Test de consistencia de datos a través del proceso."""
        dm = DataManager(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_dir=test_data_dir
        )
        
        dm.download_data()
        original_symbols = dm.symbols.copy()
        
        dm.calculate_returns()
        assert dm.symbols == original_symbols
        
        stats = dm.calculate_statistics()
        assert stats['symbols'] == original_symbols