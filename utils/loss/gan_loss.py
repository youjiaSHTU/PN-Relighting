#  # Loss function.
#     psnr_criterion        = nn.MSELoss().to(device)     # PSNR metrics.
#     pixel_criterion       = nn.MSELoss().to(device)     # Pixel loss.
#     content_criterion     = ContentLoss().to(device)    # Content loss.
#     adversarial_criterion = nn.BCELoss().to(device)     # Adversarial loss.
#     pixel_weight          = 0.01
#     content_weight        = 1.0
#     adversarial_weight    = 0.001