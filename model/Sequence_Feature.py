import torch

from torch import nn


class MyModule(nn.Module):
    def __init__(self, hidden_size, window=30, dropout=0.5):
        super(MyModule, self).__init__()
        self.loc_seq_embed = nn.Embedding(21, hidden_size, padding_idx=0)
        self.loc_seq_conv1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.loc_seq_conv2 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.loc_seq_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1, dim_feedforward=hidden_size * 4,
                                                          dropout=dropout)
        self.loc_seq_transformer = nn.TransformerEncoder(self.loc_seq_encoder, num_layers=1)

        self.loc_ss_embed = nn.Embedding(4, hidden_size, padding_idx=0)
        self.loc_ss_conv1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                          nn.BatchNorm1d(hidden_size),
                                          nn.ReLU(),
                                          nn.AvgPool1d(3, stride=1, padding=1))
        self.loc_ss_conv2 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                          nn.BatchNorm1d(hidden_size),
                                          nn.ReLU(),
                                          nn.AvgPool1d(3, stride=1, padding=1))
        self.loc_ss_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1, dim_feedforward=hidden_size * 4,
                                                         dropout=dropout)
        self.loc_ss_transformer = nn.TransformerEncoder(self.loc_ss_encoder, num_layers=1)

        self.loc_pssm_embed = nn.Sequential(nn.Conv1d(20, hidden_size, kernel_size=1),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU())
        self.loc_pssm_conv1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3, stride=1, padding=1))
        self.loc_pssm_conv2 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3, stride=1, padding=1))
        self.loc_pssm_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1,
                                                           dim_feedforward=hidden_size * 4, dropout=dropout)
        self.loc_pssm_transformer = nn.TransformerEncoder(self.loc_pssm_encoder, num_layers=1)

        self.loc_asa_embed = nn.Sequential(nn.Conv1d(1, hidden_size, kernel_size=1),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU())
        self.loc_asa_conv1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.loc_asa_conv2 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.loc_asa_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1, dim_feedforward=hidden_size * 4,
                                                          dropout=dropout)
        self.loc_asa_transformer = nn.TransformerEncoder(self.loc_asa_encoder, num_layers=1)

        self.loc_angle_embed = nn.Sequential(nn.Conv1d(4, hidden_size, kernel_size=1),
                                             nn.BatchNorm1d(hidden_size),
                                             nn.ReLU())
        self.loc_angle_conv1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                             nn.BatchNorm1d(hidden_size),
                                             nn.ReLU(),
                                             nn.AvgPool1d(3, stride=1, padding=1))
        self.loc_angle_conv2 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                             nn.BatchNorm1d(hidden_size),
                                             nn.ReLU(),
                                             nn.AvgPool1d(3, stride=1, padding=1))
        self.loc_angle_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1,
                                                            dim_feedforward=hidden_size * 4, dropout=dropout)
        self.loc_angle_transformer = nn.TransformerEncoder(self.loc_angle_encoder, num_layers=1)

        self.loc_res_embed = nn.Sequential(nn.Conv1d(11, hidden_size, kernel_size=1),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU())
        self.loc_res_conv1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.loc_res_conv2 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.loc_res_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1, dim_feedforward=hidden_size * 4,
                                                          dropout=dropout)
        self.loc_res_transformer = nn.TransformerEncoder(self.loc_res_encoder, num_layers=1)

        self.loc_mix_conv1 = nn.Sequential(nn.Conv1d(hidden_size * 6, hidden_size * 4, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size * 4),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.loc_mix_conv2 = nn.Sequential(nn.Conv1d(hidden_size * 4, hidden_size * 4, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size * 4),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.loc_mix_encoder = nn.TransformerEncoderLayer(d_model=hidden_size * 4, nhead=4,
                                                          dim_feedforward=hidden_size * 16, dropout=dropout)
        self.loc_mix_transformer = nn.TransformerEncoder(self.loc_mix_encoder, num_layers=1)

        self.glo_seq_window = nn.Sequential(nn.Conv1d(500, window, kernel_size=1),
                                            nn.ReLU())
        self.glo_seq_embed = nn.Embedding(21, hidden_size, padding_idx=0)
        self.glo_seq_conv1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.glo_seq_conv2 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.glo_seq_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1, dim_feedforward=hidden_size * 4,
                                                          dropout=dropout)
        self.glo_seq_transformer = nn.TransformerEncoder(self.glo_seq_encoder, num_layers=1)

        self.glo_ss_window = nn.Sequential(nn.Conv1d(500, window, kernel_size=1),
                                           nn.ReLU())
        self.glo_ss_embed = nn.Embedding(4, hidden_size, padding_idx=0)
        self.glo_ss_conv1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                          nn.BatchNorm1d(hidden_size),
                                          nn.ReLU(),
                                          nn.AvgPool1d(3, stride=1, padding=1))
        self.glo_ss_conv2 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                          nn.BatchNorm1d(hidden_size),
                                          nn.ReLU(),
                                          nn.AvgPool1d(3, stride=1, padding=1))
        self.glo_ss_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1, dim_feedforward=hidden_size * 4,
                                                         dropout=dropout)
        self.glo_ss_transformer = nn.TransformerEncoder(self.glo_ss_encoder, num_layers=1)

        self.glo_pssm_window = nn.Sequential(nn.Conv1d(500, window, kernel_size=1),
                                             nn.ReLU())
        self.glo_pssm_embed = nn.Sequential(nn.Conv1d(20, hidden_size, kernel_size=1),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU())
        self.glo_pssm_conv1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3, stride=1, padding=1))
        self.glo_pssm_conv2 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.AvgPool1d(3, stride=1, padding=1))
        self.glo_pssm_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1,
                                                           dim_feedforward=hidden_size * 4, dropout=dropout)
        self.glo_pssm_transformer = nn.TransformerEncoder(self.glo_pssm_encoder, num_layers=1)

        self.glo_asa_window = nn.Sequential(nn.Conv1d(500, window, kernel_size=1),
                                            nn.ReLU())
        self.glo_asa_embed = nn.Sequential(nn.Conv1d(1, hidden_size, kernel_size=1),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU())
        self.glo_asa_conv1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.glo_asa_conv2 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.glo_asa_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1, dim_feedforward=hidden_size * 4,
                                                          dropout=dropout)
        self.glo_asa_transformer = nn.TransformerEncoder(self.glo_asa_encoder, num_layers=1)

        self.glo_angle_window = nn.Sequential(nn.Conv1d(500, window, kernel_size=1),
                                              nn.ReLU())
        self.glo_angle_embed = nn.Sequential(nn.Conv1d(4, hidden_size, kernel_size=1),
                                             nn.BatchNorm1d(hidden_size),
                                             nn.ReLU())
        self.glo_angle_conv1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                             nn.BatchNorm1d(hidden_size),
                                             nn.ReLU(),
                                             nn.AvgPool1d(3, stride=1, padding=1))
        self.glo_angle_conv2 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                             nn.BatchNorm1d(hidden_size),
                                             nn.ReLU(),
                                             nn.AvgPool1d(3, stride=1, padding=1))
        self.glo_angle_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1,
                                                            dim_feedforward=hidden_size * 4, dropout=dropout)
        self.glo_angle_transformer = nn.TransformerEncoder(self.glo_angle_encoder, num_layers=1)

        self.glo_res_window = nn.Sequential(nn.Conv1d(500, window, kernel_size=1),
                                            nn.ReLU())
        self.glo_res_embed = nn.Sequential(nn.Conv1d(11, hidden_size, kernel_size=1),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU())
        self.glo_res_conv1 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.glo_res_conv2 = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.glo_res_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1, dim_feedforward=hidden_size * 4,
                                                          dropout=dropout)
        self.glo_res_transformer = nn.TransformerEncoder(self.glo_res_encoder, num_layers=1)

        self.glo_mix_conv1 = nn.Sequential(nn.Conv1d(hidden_size * 6, hidden_size * 4, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size * 4),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.glo_mix_conv2 = nn.Sequential(nn.Conv1d(hidden_size * 4, hidden_size * 4, kernel_size=7, padding=3),
                                           nn.BatchNorm1d(hidden_size * 4),
                                           nn.ReLU(),
                                           nn.AvgPool1d(3, stride=1, padding=1))
        self.glo_mix_encoder = nn.TransformerEncoderLayer(d_model=hidden_size * 4, nhead=4,
                                                          dim_feedforward=hidden_size * 16, dropout=dropout)
        self.glo_mix_transformer = nn.TransformerEncoder(self.glo_mix_encoder, num_layers=1)

        self.conv1 = nn.Sequential(nn.Conv1d(hidden_size * 10, hidden_size * 10, kernel_size=11, padding=5),
                                   nn.ReLU(),
                                   nn.AvgPool1d(kernel_size=11, stride=1, padding=5))
        self.conv2 = nn.Sequential(nn.Conv1d(hidden_size * 10, hidden_size * 10, kernel_size=13, padding=6),
                                   nn.ReLU(),
                                   nn.AvgPool1d(kernel_size=13, stride=1, padding=6))
        self.conv3 = nn.Sequential(nn.Conv1d(hidden_size * 10, hidden_size * 10, kernel_size=15, padding=7),
                                   nn.ReLU(),
                                   nn.AvgPool1d(kernel_size=15, stride=1, padding=7))

        self.l1 = nn.Sequential(nn.Linear(hidden_size * 40, hidden_size),
                                nn.ReLU())
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        loc_seq, loc_ss, loc_pssm, loc_asa, loc_angle, loc_res = x[:, :window, 0].long(), x[:, :window, 1].long(), x[:,
                                                                                                                   :window,
                                                                                                                   2:22], x[
                                                                                                                          :,
                                                                                                                          :window,
                                                                                                                          22], x[
                                                                                                                               :,
                                                                                                                               :window,
                                                                                                                               23:27], x[
                                                                                                                                       :,
                                                                                                                                       :window,
                                                                                                                                       27:]
        # seq
        loc_seq = self.loc_seq_embed(loc_seq)
        loc_seq_fea = loc_seq.permute(0, 2, 1)
        loc_seq_fea = self.loc_seq_conv1(loc_seq_fea)
        loc_seq_fea = self.loc_seq_conv2(loc_seq_fea)
        loc_seq_fea = loc_seq_fea.permute(0, 2, 1).permute(1, 0, 2)
        loc_seq_fea = self.loc_seq_transformer(loc_seq_fea)
        # ss
        loc_ss = self.loc_ss_embed(loc_ss)
        loc_ss_fea = loc_ss.permute(0, 2, 1)
        loc_ss_fea = self.loc_ss_conv1(loc_ss_fea)
        loc_ss_fea = self.loc_ss_conv2(loc_ss_fea)
        loc_ss_fea = loc_ss_fea.permute(0, 2, 1).permute(1, 0, 2)
        loc_ss_fea = self.loc_ss_transformer(loc_ss_fea)
        # pssm
        loc_pssm = loc_pssm.permute(0, 2, 1)
        loc_pssm_fea = self.loc_pssm_embed(loc_pssm)
        loc_pssm = loc_pssm_fea.permute(0, 2, 1)
        loc_pssm_fea = self.loc_pssm_conv1(loc_pssm_fea)
        loc_pssm_fea = self.loc_pssm_conv2(loc_pssm_fea)
        loc_pssm_fea = loc_pssm_fea.permute(0, 2, 1).permute(1, 0, 2)
        loc_pssm_fea = self.loc_pssm_transformer(loc_pssm_fea)
        # asa
        loc_asa = loc_asa.unsqueeze(2).permute(0, 2, 1)
        loc_asa_fea = self.loc_asa_embed(loc_asa)
        loc_asa = loc_asa_fea.permute(0, 2, 1)
        loc_asa_fea = self.loc_asa_conv1(loc_asa_fea)
        loc_asa_fea = self.loc_asa_conv2(loc_asa_fea)
        loc_asa_fea = loc_asa_fea.permute(0, 2, 1).permute(1, 0, 2)
        loc_asa_fea = self.loc_asa_transformer(loc_asa_fea)
        # angle
        loc_angle = loc_angle.permute(0, 2, 1)
        loc_angle_fea = self.loc_angle_embed(loc_angle)
        loc_angle = loc_angle_fea.permute(0, 2, 1)
        loc_angle_fea = self.loc_angle_conv1(loc_angle_fea)
        loc_angle_fea = self.loc_angle_conv2(loc_angle_fea)
        loc_angle_fea = loc_angle_fea.permute(0, 2, 1).permute(1, 0, 2)
        loc_angle_fea = self.loc_angle_transformer(loc_angle_fea)
        # res
        loc_res = loc_res.permute(0, 2, 1)
        loc_res_fea = self.loc_res_embed(loc_res)
        loc_res = loc_res_fea.permute(0, 2, 1)
        loc_res_fea = self.loc_res_conv1(loc_res_fea)
        loc_res_fea = self.loc_res_conv2(loc_res_fea)
        loc_res_fea = loc_res_fea.permute(0, 2, 1).permute(1, 0, 2)
        loc_res_fea = self.loc_res_transformer(loc_res_fea)
        # mix
        loc_mix = torch.cat((loc_seq, loc_ss, loc_pssm, loc_asa, loc_angle, loc_res), -1)
        loc_mix_fea = loc_mix.permute(0, 2, 1)
        loc_mix_fea = self.loc_mix_conv1(loc_mix_fea)
        loc_mix_fea = self.loc_mix_conv2(loc_mix_fea)
        loc_mix_fea = loc_mix_fea.permute(0, 2, 1).permute(1, 0, 2)
        loc_mix_fea = self.loc_mix_transformer(loc_mix_fea)

        glo_seq, glo_ss, glo_pssm, glo_asa, glo_angle, glo_res = x[:, window:, 0].long(), x[:, window:, 1].long(), x[:,
                                                                                                                   window:,
                                                                                                                   2:22], x[
                                                                                                                          :,
                                                                                                                          window:,
                                                                                                                          22], x[
                                                                                                                               :,
                                                                                                                               window:,
                                                                                                                               23:27], x[
                                                                                                                                       :,
                                                                                                                                       window:,
                                                                                                                                       27:]
        # seq
        glo_seq = self.glo_seq_embed(glo_seq)
        glo_seq = self.glo_seq_window(glo_seq)
        glo_seq_fea = glo_seq.permute(0, 2, 1)
        glo_seq_fea = self.glo_seq_conv1(glo_seq_fea)
        glo_seq_fea = self.glo_seq_conv2(glo_seq_fea)
        glo_seq_fea = glo_seq_fea.permute(0, 2, 1).permute(1, 0, 2)
        glo_seq_fea = self.glo_seq_transformer(glo_seq_fea)
        # ss
        glo_ss = self.glo_ss_embed(glo_ss)
        glo_ss = self.glo_ss_window(glo_ss)
        glo_ss_fea = glo_ss.permute(0, 2, 1)
        glo_ss_fea = self.glo_ss_conv1(glo_ss_fea)
        glo_ss_fea = self.glo_ss_conv2(glo_ss_fea)
        glo_ss_fea = glo_ss_fea.permute(0, 2, 1).permute(1, 0, 2)
        glo_ss_fea = self.glo_ss_transformer(glo_ss_fea)
        # pssm
        glo_pssm = self.glo_pssm_window(glo_pssm).permute(0, 2, 1)
        glo_pssm_fea = self.glo_pssm_embed(glo_pssm)
        glo_pssm = glo_pssm_fea.permute(0, 2, 1)
        glo_pssm_fea = self.glo_pssm_conv1(glo_pssm_fea)
        glo_pssm_fea = self.glo_pssm_conv2(glo_pssm_fea)
        glo_pssm_fea = glo_pssm_fea.permute(0, 2, 1).permute(1, 0, 2)
        glo_pssm_fea = self.glo_pssm_transformer(glo_pssm_fea)
        # asa
        glo_asa = glo_asa.unsqueeze(2)
        glo_asa = self.glo_asa_window(glo_asa).permute(0, 2, 1)
        glo_asa_fea = self.glo_asa_embed(glo_asa)
        glo_asa = glo_asa_fea.permute(0, 2, 1)
        glo_asa_fea = self.glo_asa_conv1(glo_asa_fea)
        glo_asa_fea = self.glo_asa_conv2(glo_asa_fea)
        glo_asa_fea = glo_asa_fea.permute(0, 2, 1).permute(1, 0, 2)
        glo_asa_fea = self.glo_asa_transformer(glo_asa_fea)
        # angle
        glo_angle = self.glo_angle_window(glo_angle).permute(0, 2, 1)
        glo_angle_fea = self.glo_angle_embed(glo_angle)
        glo_angle = glo_angle_fea.permute(0, 2, 1)
        glo_angle_fea = self.glo_angle_conv1(glo_angle_fea)
        glo_angle_fea = self.glo_angle_conv2(glo_angle_fea)
        glo_angle_fea = glo_angle_fea.permute(0, 2, 1).permute(1, 0, 2)
        glo_angle_fea = self.glo_angle_transformer(glo_angle_fea)
        # res
        glo_res = self.glo_res_window(glo_res).permute(0, 2, 1)
        glo_res_fea = self.glo_res_embed(glo_res)
        glo_res = glo_res_fea.permute(0, 2, 1)
        glo_res_fea = self.glo_res_conv1(glo_res_fea)
        glo_res_fea = self.glo_res_conv2(glo_res_fea)
        glo_res_fea = glo_res_fea.permute(0, 2, 1).permute(1, 0, 2)
        glo_res_fea = self.glo_res_transformer(glo_res_fea)
        # mix
        glo_mix = torch.cat((glo_seq, glo_ss, glo_pssm, glo_asa, glo_angle, glo_res), -1)
        glo_mix_fea = glo_mix.permute(0, 2, 1)
        glo_mix_fea = self.glo_mix_conv1(glo_mix_fea)
        glo_mix_fea = self.glo_mix_conv2(glo_mix_fea)
        glo_mix_fea = glo_mix_fea.contiguous().permute(0, 2, 1).permute(1, 0, 2)
        glo_mix_fea = self.glo_mix_transformer(glo_mix_fea)

        loc_fea = torch.cat(
            (loc_mix_fea, loc_seq_fea, loc_ss_fea, loc_pssm_fea, loc_asa_fea, loc_angle_fea, loc_res_fea), -1).permute(
            1, 0, 2)
        glo_fea = torch.cat(
            (glo_mix_fea, glo_seq_fea, glo_ss_fea, glo_pssm_fea, glo_asa_fea, glo_angle_fea, glo_res_fea), -1).permute(
            1, 0, 2)
        glo_fea = glo_fea.permute(0, 2, 1)
        glo_fea1 = self.conv1(glo_fea)
        glo_fea2 = self.conv2(glo_fea)
        glo_fea3 = self.conv3(glo_fea)
        glo_fea1, glo_fea2, glo_fea3 = glo_fea1.permute(0, 2, 1), glo_fea2.permute(0, 2, 1), glo_fea3.permute(0, 2, 1)

        fea = torch.cat((loc_fea, glo_fea1, glo_fea2, glo_fea3), -1)
        fea = self.l1(fea)
        res = self.l2(fea)

        return res


window = 30


class Sequence_Ex(nn.Module):
    def __init__(self, hidden_size=128, drop_out=0.1):
        super(Sequence_Ex, self).__init__()

        self.drop_out = drop_out
        self.hidden_size = 128
        self.local_reduce_dim = nn.Linear(1024, self.hidden_size)
        self.global_reduce_dim = nn.Linear(1024, self.hidden_size)
        self.loc_seq_encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size * 4,
                                                          dropout=drop_out)
        self.loc_seq_transformer = nn.TransformerEncoder(self.loc_seq_encoder, num_layers=6)

        self.local_conv1 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.AvgPool1d(3, stride=3, padding=1))

        self.local_conv2 = nn.Sequential(
            nn.Conv1d(hidden_size * 2, hidden_size * 2 * 2, kernel_size=3),
            nn.BatchNorm1d(hidden_size * 2 * 2),
            nn.ReLU(),
            nn.AvgPool1d(3, stride=3, padding=1))

        self.local_reduce_dim_2=nn.Linear(512*2,256)
        return

    def forward(self, local_sequence):
        'local=L*20*1024'
        'global_sequence=L*1024'

        local_sequence = self.local_reduce_dim(local_sequence)
        local_sequence = self.loc_seq_transformer(local_sequence)
        local_sequence = self.local_conv1(local_sequence)
        local_sequence = self.local_conv2(local_sequence)
        local_sequence = local_sequence.reshape(local_sequence.shape[0],
                                                local_sequence.shape[1] * local_sequence.shape[2])
        local_sequence=self.local_reduce_dim_2(local_sequence)
        'L*256'

        return local_sequence
