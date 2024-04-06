import pytest
import torch
from pytorch_custom.pytorch_base import BaseModel

class TestBaseModel:

    @pytest.fixture
    def model(self):
        return BaseModel(10, 20, 3)

    @pytest.mark.parametrize("input_size, hidden_size, output_size", [
        (5, 10, 2),
        (15, 30, 4),
    ])
    def test_initialization(self, input_size, hidden_size, output_size):
        model = BaseModel(input_size, hidden_size, output_size)
        assert model.linear1.in_features == input_size
        assert model.linear1.out_features == hidden_size
        assert model.linear2.in_features == hidden_size
        assert model.linear2.out_features == output_size

    def test_forward_pass(self, model):
        input_tensor = torch.randn(4, 10)  # Batch of 4 samples, each with 10 features
        output_tensor = model(input_tensor)
        assert output_tensor.shape == (4, 3)

    def test_loss_calculation(self, model):
        outputs = torch.randn(4, 3)
        targets = torch.randn(4, 3)
        loss_fn = torch.nn.MSELoss()
        loss = model.calculate_loss(outputs, targets, loss_fn)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])

    # @pytest.mark.parametrize("input_size, hidden_size, output_size, expected_exception", [
    #     (-1, 10, 3, ValueError),  # Assuming a ValueError for invalid input_size
    #     (10, 0, 3, RuntimeError),  # Assuming a RuntimeError for invalid hidden_size
    #     (10, 20, -2, TypeError),  # Assuming a TypeError for invalid output_size
    # ])
    # def test_invalid_initialization(self, input_size, hidden_size, output_size, expected_exception):
    #     with pytest.raises(expected_exception):
    #         BaseModel(input_size, hidden_size, output_size)
            