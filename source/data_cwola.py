import torch


def split_by_pure_random(
    sig_tensor: torch.Tensor, bkg_tensor: torch.Tensor,
    num_train: int, num_valid: int, num_test: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    print(f"{'=' * 20} Data Size Information {'=' * 20}")
    print(f'Signal shape: {sig_tensor.shape}, Background shape: {bkg_tensor.shape}')

    # Randomly sampling and artificially set num_test for signal and background
    sig_index = torch.randperm(len(sig_tensor))
    bkg_index = torch.randperm(len(bkg_tensor))
    sig_train_index = sig_index[:num_train]
    bkg_train_index = bkg_index[:num_train]
    sig_valid_index = sig_index[num_train:num_train + num_valid]
    bkg_valid_index = bkg_index[num_train:num_train + num_valid]
    sig_test_index = sig_index[num_train + num_valid:num_train + num_valid + num_test]
    bkg_test_index = bkg_index[num_train + num_valid:num_train + num_valid + num_test]

    # Create mixed tensors for implementing CWoLa
    train_sig = sig_tensor[sig_train_index]
    train_bkg = bkg_tensor[bkg_train_index]
    valid_sig = sig_tensor[sig_valid_index]
    valid_bkg = bkg_tensor[bkg_valid_index]
    test_sig = sig_tensor[sig_test_index]
    test_bkg = bkg_tensor[bkg_test_index]

    print(f'Train signal shape: {train_sig.shape}, Train background shape: {train_bkg.shape}')
    print(f'Valid signal shape: {valid_sig.shape}, Valid background shape: {valid_bkg.shape}')
    print(f'Test signal shape: {test_sig.shape}, Test background shape: {test_bkg.shape}')
    print(f"{'=' * 50}")

    return train_sig, train_bkg, valid_sig, valid_bkg, test_sig, test_bkg


def split_by_jet_flavor(
    sig_tensor: torch.Tensor, bkg_tensor: torch.Tensor,
    sig_flavor: dict[str, torch.Tensor], bkg_flavor: dict[str, torch.Tensor],
    branching_ratio: float, luminosity: float,
    sig_cross_section: float, bkg_cross_section: float,
    sig_preselection_rate: float = 1, bkg_preselection_rate: float = 1,
    train_fraction: float = 0.8, num_test: int = 10000
) -> tuple[torch.Tensor]:

    print(f"{'=' * 20} Data Size Information {'=' * 20}")
    print(f'Signal shape: {sig_tensor.shape}, Background shape: {bkg_tensor.shape}')

    # Number of data from real luminosity after preselection
    num_sig_preselection = luminosity * sig_cross_section * branching_ratio * sig_preselection_rate
    num_bkg_preselection = luminosity * bkg_cross_section * branching_ratio * bkg_preselection_rate

    # Additional selection: only keep 2q0g, 1q1g, and 0q2g events
    num_sig = int((sum(sig_flavor['2q0g']) + sum(sig_flavor['1q1g']) + sum(sig_flavor['0q2g'])) / sig_flavor['total'] * num_sig_preselection)
    num_bkg = int((sum(bkg_flavor['2q0g']) + sum(bkg_flavor['1q1g']) + sum(bkg_flavor['0q2g'])) / bkg_flavor['total'] * num_bkg_preselection)
    print(f'Signal after selection: {num_sig}, Background after selection: {num_bkg}')

    # Indices of 2q0g, 1q1g, and 0q2g events after selection
    sig_index = torch.nonzero((sig_flavor['2q0g'] | sig_flavor['1q1g'] | sig_flavor['0q2g'])).squeeze()
    bkg_index = torch.nonzero((bkg_flavor['2q0g'] | bkg_flavor['1q1g'] | bkg_flavor['0q2g'])).squeeze()

    # Randomly sampling and artificially set num_test for signal and background
    sig_index = sig_index[torch.randperm(len(sig_index))]
    bkg_index = bkg_index[torch.randperm(len(bkg_index))]
    sig_train_index = sig_index[:int(num_sig * train_fraction)]
    bkg_train_index = bkg_index[:int(num_bkg * train_fraction)]
    sig_valid_index = sig_index[int(num_sig * train_fraction):int(num_sig)]
    bkg_valid_index = bkg_index[int(num_bkg * train_fraction):int(num_bkg)]
    sig_test_index = sig_index[int(num_sig):int(num_sig) + num_test]
    bkg_test_index = bkg_index[int(num_bkg):int(num_bkg) + num_test]

    # Create mixed tensors for implementing CWoLa
    train_sig = torch.cat((
        sig_tensor[sig_train_index][sig_flavor['2q0g'][sig_train_index]],
        bkg_tensor[bkg_train_index][bkg_flavor['2q0g'][bkg_train_index]]
    ), dim=0)
    train_bkg = torch.cat((
        sig_tensor[sig_train_index][sig_flavor['1q1g'][sig_train_index] | sig_flavor['0q2g'][sig_train_index]],
        bkg_tensor[bkg_train_index][bkg_flavor['1q1g'][bkg_train_index] | bkg_flavor['0q2g'][bkg_train_index]]
    ), dim=0)
    valid_sig = torch.cat((
        sig_tensor[sig_valid_index][sig_flavor['2q0g'][sig_valid_index]],
        bkg_tensor[bkg_valid_index][bkg_flavor['2q0g'][bkg_valid_index]]
    ), dim=0)
    valid_bkg = torch.cat((
        sig_tensor[sig_valid_index][sig_flavor['1q1g'][sig_valid_index] | sig_flavor['0q2g'][sig_valid_index]],
        bkg_tensor[bkg_valid_index][bkg_flavor['1q1g'][bkg_valid_index] | bkg_flavor['0q2g'][bkg_valid_index]]
    ), dim=0)
    test_sig = sig_tensor[sig_test_index]
    test_bkg = bkg_tensor[bkg_test_index]
    print(f'Train signal shape: {train_sig.shape}, Train background shape: {train_bkg.shape}')
    print(f'Valid signal shape: {valid_sig.shape}, Valid background shape: {valid_bkg.shape}')
    print(f'Test signal shape: {test_sig.shape}, Test background shape: {test_bkg.shape}')
    print(f"{'=' * 50}")

    return train_sig, train_bkg, valid_sig, valid_bkg, test_sig, test_bkg
