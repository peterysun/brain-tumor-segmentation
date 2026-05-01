import torch
import torch.nn as nn


class UNet(nn.Module):

    """
    There will be 3 channels for each RGB color within the image.
    The model will check a 4x4 grid within the RGB
    And for each 4x4 grid, there will be 3 copies of them for each color
    The filter acts as a sliding window of a 3x3 for example.
    It will scan a 3x3 section within the 4x4 or say 6x6, and
    there will be weights attached to each filter. 
    Say [-1 0 1] per row in the 3x3 so -1 in u1, 0 in u2, 1 in u3
    These weights will be learnable weights that adjust after each
    echon. The dot product between the weights and the channel
    values within each 3x3 section of the 4x4 or 6x6 will be taken.
    Then there will be a total for each position of the window. So for
    a 4x4 there would end up to be a 2x2 of total values per window (4 values)
    and a 6x6 would have 16 total values per window. The reason the
    windows are sliding is because the code needs to check the edges of each
    window. If there is a big edge change such as u1 = 10 to u2 = 200 and u3 = 200,
    then we know that there is a edge between u1 and u2. This means that there
    is most likely a big change in color within the image in that section,
    showing there is a potential tumor. You then take each value across the 3
    channels and sum them together for one total value. Then you would take all the
    total values and create an array. So a 256x256 would turn into a 254x254 in the sliding window
    method because there are 4 sides or edges of the entire 256x256 array. You
    always would lose a border weight of 2 pixels with a 3x3 filter. 1 per side.
    Then with padding it goes from 254x254 back to 256 using 0's on the side lines.
    In our case there would be 64 filters. So you do that 64 times total.
    
    
    """

    def __init__(self):
        super(UNet, self).__init__()


        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)


        self.bottleneck = self.conv_block(512, 1024)


        self.upconv4 = self.upconv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)

        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)

        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)

        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)


        self.final = nn.Conv2d(64, 1, kernel_size=1)


        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        """
        Upsampling: makes the image bigger again.
        ConvTranspose2d is like a reverse convolution.
        """
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):



        e1 = self.enc1(x)  # (3, 256, 256)   → (64, 256, 256)
        e2 = self.enc2(self.pool(e1))  # (64, 128, 128)  → (128, 128, 128)
        e3 = self.enc3(self.pool(e2))  # (128, 64, 64)   → (256, 64, 64)
        e4 = self.enc4(self.pool(e3))  # (256, 32, 32)   → (512, 32, 32)


        b = self.bottleneck(self.pool(e4))  # (512, 16, 16) → (1024, 16, 16)


        d4 = self.upconv4(b)  # (1024, 16, 16) → (512, 32, 32)
        d4 = torch.cat([d4, e4], dim=1)  # concat with encoder: (1024, 32, 32)
        d4 = self.dec4(d4)  # (1024, 32, 32) → (512, 32, 32)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Final output
        out = self.final(d1)  # (64, 256, 256) → (1, 256, 256)
        out = torch.sigmoid(out)  #  0-1

        return out


#test
if __name__ == "__main__":
    model = UNet()


    fake_input = torch.randn(2, 3, 256, 256)


    output = model(fake_input)

    print(f"Input shape:  {fake_input.shape}")  # (2, 3, 256, 256)
    print(f"Output shape: {output.shape}")  # (2, 1, 256, 256)
    print(f"Output range: {output.min():.3f} to {output.max():.3f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
