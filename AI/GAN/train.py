from GAN import *
from dataset import parse_dataset

def trainer():
    x_train = parse_dataset()[0][0]
    
    dset_len = {"train": len(train_dsets)}
    dset_loaders = {"train": train_dset_loaders}
    # print (dset_len["train"])

    # model define
    D = net_D(args)
    G = net_G(args)


    D_solver = optim.Adam(D.parameters(), lr=params.d_lr, betas=params.beta)
    # D_solver = optim.SGD(D.parameters(), lr=args.d_lr, momentum=0.9)
    G_solver = optim.Adam(G.parameters(), lr=params.g_lr, betas=params.beta)

    D.to(params.device)
    G.to(params.device)

    # criterion_D = nn.BCELoss()
    criterion_D = nn.MSELoss()

    criterion_G = nn.L1Loss()

    itr_val = -1
    itr_train = -1

    for epoch in range(params.epochs):

        start = time.time()

        for phase in ['train']:
            if phase == 'train':
                # if args.lrsh:
                #     D_scheduler.step()
                D.train()
                G.train()
            else:
                D.eval()
                G.eval()

            running_loss_G = 0.0
            running_loss_D = 0.0
            running_loss_adv_G = 0.0

            for i, X in enumerate(tqdm(dset_loaders[phase])):

                # if phase == 'val':
                #     itr_val += 1

                if phase == 'train':
                    itr_train += 1

                X = X.to(params.device)
                # print (X)
                # print (X.size())

                batch = X.size()[0]
                # print (batch)

                Z = generateZ(args, batch)
                # print (Z.size())

                # ============= Train the discriminator =============#
                d_real = D(X)

                fake = G(Z)
                d_fake = D(fake)

                real_labels = torch.ones_like(d_real).to(params.device)
                fake_labels = torch.zeros_like(d_fake).to(params.device)
                # print (d_fake.size(), fake_labels.size())

                if params.soft_label:
                    real_labels = torch.Tensor(batch).uniform_(0.7, 1.2).to(params.device)
                    fake_labels = torch.Tensor(batch).uniform_(0, 0.3).to(params.device)

                # print (d_real.size(), real_labels.size())
                d_real_loss = criterion_D(d_real, real_labels)

                d_fake_loss = criterion_D(d_fake, fake_labels)

                d_loss = d_real_loss + d_fake_loss

                # no deleted
                d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
                d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
                d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

                if d_total_acu < params.d_thresh:
                    D.zero_grad()
                    d_loss.backward()
                    D_solver.step()

                # =============== Train the generator ===============#

                Z = generateZ(args, batch)

                # print (X)
                fake = G(Z)  # generated fake: 0-1, X: 0/1
                d_fake = D(fake)

                adv_g_loss = criterion_D(d_fake, real_labels)
                # print (fake.size(), X.size())

                # recon_g_loss = criterion_D(fake, X)
                recon_g_loss = criterion_G(fake, X)
                # g_loss = recon_g_loss + params.adv_weight * adv_g_loss
                g_loss = adv_g_loss

        
                D.zero_grad()
                G.zero_grad()
                g_loss.backward()
                G_solver.step()

                # =============== logging each 10 iterations ===============#

                running_loss_G += recon_g_loss.item() * X.size(0)
                running_loss_D += d_loss.item() * X.size(0)
                running_loss_adv_G += adv_g_loss.item() * X.size(0)


            epoch_loss_D = running_loss_D / dset_len[phase]
            epoch_loss_adv_G = running_loss_adv_G / dset_len[phase]

            print('Epochs-{} ({}) , D(x) : {:.4}, D(G(x)) : {:.4}'.format(epoch, phase, epoch_loss_D, epoch_loss_adv_G))
