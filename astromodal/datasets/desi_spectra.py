import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def stitch_desi_arms(
    wave_b, flux_b, ivar_b,
    wave_r, flux_r, ivar_r,
    wave_z, flux_z, ivar_z,
):
    """
    Stitch DESI b/r/z arms into a single spectrum.

    Returns:
      wave, flux, ivar, mask

    mask:
      - True  : valid spectral pixel
      - False : invalid (ivar == 0 or missing)
    """
    wb = np.asarray(wave_b, dtype=np.float64)
    fb = np.asarray(flux_b, dtype=np.float64)
    ivb = np.asarray(ivar_b, dtype=np.float64)

    wr = np.asarray(wave_r, dtype=np.float64)
    fr = np.asarray(flux_r, dtype=np.float64)
    ivr = np.asarray(ivar_r, dtype=np.float64)

    wz = np.asarray(wave_z, dtype=np.float64)
    fz = np.asarray(flux_z, dtype=np.float64)
    ivz = np.asarray(ivar_z, dtype=np.float64)

    # concatenate
    w = np.concatenate([wb, wr, wz])
    f = np.concatenate([fb, fr, fz])
    iv = np.concatenate([ivb, ivr, ivz])

    # finite + non-negative ivar
    m = np.isfinite(w) & np.isfinite(f) & np.isfinite(iv) & (iv >= 0)
    w, f, iv = w[m], f[m], iv[m]

    if w.size == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=bool),
        )

    # sort by wavelength
    order = np.argsort(w)
    w, f, iv = w[order], f[order], iv[order]

    # combine duplicate wavelengths (arm overlaps)
    w_unique, inv, counts = np.unique(w, return_inverse=True, return_counts=True)

    if np.any(counts > 1):
        num = np.zeros_like(w_unique, dtype=np.float64)
        den = np.zeros_like(w_unique, dtype=np.float64)
        np.add.at(num, inv, iv * f)
        np.add.at(den, inv, iv)

        flux_out = np.zeros_like(w_unique, dtype=np.float64)
        mask_out = den > 0

        # ivar-weighted mean where possible
        flux_out[mask_out] = num[mask_out] / den[mask_out]

        # fallback to simple mean if all ivar == 0
        bad = ~mask_out
        if np.any(bad):
            sum_f = np.zeros_like(w_unique, dtype=np.float64)
            np.add.at(sum_f, inv, f)
            flux_out[bad] = sum_f[bad] / counts[bad]

        ivar_out = den
        return w_unique, flux_out, ivar_out, mask_out

    # no duplicates
    mask = iv > 0
    return w, f, iv, mask

class DesiStitchedSpectraDataset(Dataset):
    """
    Expects a Polars DataFrame (or any object that supports .row(i, named=True))
    with columns:
      desi_wave_b, desi_flux_b, desi_ivar_b,
      desi_wave_r, desi_flux_r, desi_ivar_r,
      desi_wave_z, desi_flux_z, desi_ivar_z

    Returns per item:
      x    : torch.float32 [L, 2]   (flux, ivar)
      mask : torch.bool    [L]
    """
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return self.df.height

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)

        w, f, iv, m = stitch_desi_arms(
            row["desi_wave_b"], row["desi_flux_b"], row["desi_ivar_b"],
            row["desi_wave_r"], row["desi_flux_r"], row["desi_ivar_r"],
            row["desi_wave_z"], row["desi_flux_z"], row["desi_ivar_z"],
        )

        x = np.stack([f.astype(np.float32), iv.astype(np.float32)], axis=1)  # [L,2]
        return torch.from_numpy(x), torch.from_numpy(m.astype(np.bool_))


def desi_collate_pad(batch, pad_value_x=0.0, pad_value_mask=False):
    """
    Pads variable-length stitched spectra in a batch.

    batch: list of (x[L,2], mask[L])
    returns:
      x_pad    : [B, Lmax, 2] float32
      mask_pad : [B, Lmax]    bool
      lengths  : [B]          int64
    """
    xs, ms = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    Lmax = int(lengths.max().item())
    B = len(xs)

    x_pad = torch.full((B, Lmax, 2), float(pad_value_x), dtype=torch.float32)
    mask_pad = torch.full((B, Lmax), bool(pad_value_mask), dtype=torch.bool)

    for i, (x, m) in enumerate(zip(xs, ms)):
        L = x.shape[0]
        x_pad[i, :L] = x
        mask_pad[i, :L] = m

    return x_pad, mask_pad, lengths


# Example usage:
# dataset = DesiStitchedSpectraDataset(df)
# loader = DataLoader(
#     dataset,
#     batch_size=64,
#     shuffle=True,
#     num_workers=8,
#     pin_memory=True,
#     collate_fn=desi_collate_pad,
# )