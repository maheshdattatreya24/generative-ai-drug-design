# =========================================================
# COMPLETE DRUG GENERATION PIPELINE
# SMILES → SELFIES → VAE → TRANSFORMER → GAN → EVALUATION
# =========================================================

# ========================
# STEP 0: SETUP
# ========================
from google.colab import drive
drive.mount('/content/drive')

!pip install selfies -q

import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Crippen, Lipinski, Draw
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("✅ Device:", device)

# ========================
# STEP 1: DATA
# ========================
url = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"
df = pd.read_csv(url)

smiles = df['SMILES'].dropna().tolist()[:20000]

# SMILES → SELFIES
selfies_data = []
for s in smiles:
    try:
        selfies_data.append(sf.encoder(s))
    except:
        continue

print("✅ SELFIES dataset:", len(selfies_data))

# ========================
# TOKENIZER
# ========================
alphabet = list(sf.get_alphabet_from_selfies(selfies_data))
alphabet = ['[PAD]', '[SOS]', '[EOS]'] + alphabet

stoi = {s:i for i,s in enumerate(alphabet)}
itos = {i:s for s,i in stoi.items()}

def encode(s, max_len=60):
    tokens = [stoi['[SOS]']]
    for tok in sf.split_selfies(s):
        tokens.append(stoi.get(tok,0))
    tokens.append(stoi['[EOS]'])
    tokens = tokens[:max_len]
    tokens += [0]*(max_len-len(tokens))
    return tokens

def decode(tokens):
    res=[]
    for t in tokens:
        if t==stoi['[EOS]']: break
        if t not in [0,stoi['[SOS]']]:
            res.append(itos[t])
    return "".join(res)

vocab_size = len(alphabet)

# ========================
# DATASET
# ========================
class MolDataset(Dataset):
    def __init__(self,data): self.data=data
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        return torch.tensor(encode(self.data[i]))

loader = DataLoader(MolDataset(selfies_data), batch_size=64, shuffle=True)

# ========================
# VAE
# ========================
class VAE(nn.Module):
    def __init__(self,vocab):
        super().__init__()
        self.emb=nn.Embedding(vocab,64)
        self.enc=nn.GRU(64,128,batch_first=True)
        self.mu=nn.Linear(128,64)
        self.logvar=nn.Linear(128,64)
        self.dec=nn.GRU(64+64,128,batch_first=True)
        self.out=nn.Linear(128,vocab)

    def forward(self,x):
        e=self.emb(x)
        _,h=self.enc(e)
        mu,logvar=self.mu(h[-1]),self.logvar(h[-1])
        z=mu+torch.randn_like(mu)*torch.exp(0.5*logvar)

        z_exp=z.unsqueeze(1).repeat(1,x.size(1),1)
        d_in=torch.cat([self.emb(x),z_exp],dim=2)
        out,_=self.dec(d_in)
        return self.out(out),mu,logvar

vae = VAE(vocab_size).to(device)
opt = torch.optim.Adam(vae.parameters(),1e-3)

print("\n🧠 Training VAE...")
for epoch in range(5):
    total=0
    for batch in loader:
        batch=batch.to(device)
        opt.zero_grad()

        logits,mu,logvar = vae(batch)
        recon = F.cross_entropy(logits.view(-1,vocab_size),
                                batch.view(-1),ignore_index=0)
        kl = -0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())

        loss = recon + 0.1*kl
        loss.backward()
        opt.step()
        total+=loss.item()

    print(f"VAE Epoch {epoch+1} Loss: {total/len(loader):.4f}")

# ========================
# TRANSFORMER
# ========================
class Transformer(nn.Module):
    def __init__(self,vocab):
        super().__init__()
        self.emb=nn.Embedding(vocab,128)
        self.tr=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128,4,256,batch_first=True),2)
        self.fc=nn.Linear(128,vocab)

    def forward(self,x):
        mask=torch.triu(torch.ones(x.size(1),x.size(1)),1).bool().to(device)
        return self.fc(self.tr(self.emb(x),mask=mask))

    def generate(self,n=100):
        self.eval()
        res=[]
        with torch.no_grad():
            for _ in range(n):
                x=torch.tensor([[stoi['[SOS]']]],device=device)
                for _ in range(60):
                    logits=self(x)[:,-1,:]
                    probs=torch.softmax(logits,dim=-1)
                    nxt=torch.multinomial(probs,1)
                    x=torch.cat([x,nxt],1)
                res.append(decode(x[0].cpu().tolist()))
        return res

tr_model = Transformer(vocab_size).to(device)
opt = torch.optim.Adam(tr_model.parameters(),1e-3)

print("\n🤖 Training Transformer...")
for epoch in range(6):
    total=0
    for batch in loader:
        batch=batch.to(device)
        opt.zero_grad()

        out=tr_model(batch)
        loss=F.cross_entropy(out[:,:-1].reshape(-1,vocab_size),
                             batch[:,1:].reshape(-1),ignore_index=0)
        loss.backward()
        opt.step()
        total+=loss.item()

    print(f"Transformer Epoch {epoch+1} Loss: {total/len(loader):.4f}")

# ========================
# GAN
# ========================
class Discriminator(nn.Module):
    def __init__(self,vocab):
        super().__init__()
        self.emb=nn.Embedding(vocab,64)
        self.conv=nn.Sequential(
            nn.Conv1d(64,128,3,padding=1),
            nn.ReLU(),
            nn.Conv1d(128,256,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc=nn.Linear(256,1)

    def forward(self,x):
        x=self.emb(x).transpose(1,2)
        x=self.conv(x).squeeze(-1)
        return torch.sigmoid(self.fc(x))

disc = Discriminator(vocab_size).to(device)
d_opt = torch.optim.Adam(disc.parameters(),2e-4)
g_opt = torch.optim.Adam(tr_model.parameters(),1e-4)
criterion = nn.BCELoss()

print("\n🧪 Training GAN...")
for epoch in range(3):
    for batch in loader:
        batch=batch.to(device)
        bs=batch.size(0)

        real_labels = torch.ones(bs,1).to(device)
        fake_labels = torch.zeros(bs,1).to(device)

        # Train Discriminator
        d_opt.zero_grad()
        real_out = disc(batch)
        d_real_loss = criterion(real_out, real_labels)

        fake_selfies = tr_model.generate(bs)
        fake_encoded = torch.tensor([encode(s) for s in fake_selfies],
                                    dtype=torch.long, device=device)

        fake_out = disc(fake_encoded)
        d_fake_loss = criterion(fake_out, fake_labels)

        d_loss = (d_real_loss + d_fake_loss)/2
        d_loss.backward()
        d_opt.step()

        # Train Generator
        g_opt.zero_grad()
        fake_out = disc(fake_encoded)
        g_loss = criterion(fake_out, real_labels)
        g_loss.backward()
        g_opt.step()

    print(f"GAN Epoch {epoch+1}")

# ========================
# EVALUATION
# ========================
print("\n🔬 Evaluating...")

gen_selfies = tr_model.generate(100)

gen_smiles = []
for s in gen_selfies:
    try:
        gen_smiles.append(sf.decoder(s))
    except:
        continue

results=[]
for smi in gen_smiles:
    mol=Chem.MolFromSmiles(smi)
    if mol:
        results.append({
            "SMILES": smi,
            "QED": QED.qed(mol),
            "MW": Descriptors.MolWt(mol),
            "LogP": Crippen.MolLogP(mol)
        })

df = pd.DataFrame(results)

print("\n📊 RESULTS")
print(f"Generated: {len(gen_smiles)}")
print(f"Valid: {len(df)}")
print(f"Validity: {(len(df)/len(gen_smiles))*100:.2f}%")

print("\n🧾 SAMPLE:")
for i,row in df.head(10).iterrows():
    print(row["SMILES"])

# ========================
# VISUALIZATION
# ========================
plt.hist(df['QED'], bins=20)
plt.title("QED Distribution")
plt.show()

mols = [Chem.MolFromSmiles(s) for s in df.head(10)['SMILES']]
img = Draw.MolsToGridImage(mols, molsPerRow=5)

from IPython.display import display
display(img)

print("\n🎉 COMPLETE PIPELINE DONE")
# ========================# STEP 8: FINAL EVALUATION + VISUALIZATION (ENHANCED)# ========================# Define SAVE_DIR for saving results to Google Drive
SAVE_DIR = '/content/drive/MyDrive'

from rdkit.Chem import Draw
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO # Added for robust image saving
from PIL import Image as PILImage # Added for robust image saving

print("\n" + "="*60)
print("🚀 FINAL PIPELINE EXECUTION STARTED")
print("="*60)

print("\n🔬 Generating molecules using trained Transformer...")
generated_selfies = tr_model.generate(200) # Renamed 'gen' to 'generated_selfies' for clarity

print(f"✅ Total molecules generated (SELFIES): {len(generated_selfies)}")

# Convert generated SELFIES to SMILES
print("\nConverting SELFIES to SMILES and validating...")
generated_smiles = []
for s_selfie in generated_selfies:
    try:
        s_smiles = sf.decoder(s_selfie)
        generated_smiles.append(s_smiles)
    except:
        continue

print(f"✅ Successfully converted {len(generated_smiles)} SELFIES to SMILES.")

# ========================
# VALIDATION
# ========================
print("\n🧪 Validating molecules using RDKit...")

valid_smiles = []
qed_scores = []
mw_values = []
logp_values = []

# Iterate over the *generated_smiles* list, not the original 'gen' (generated_selfies)
for smi in generated_smiles:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol: # Check if MolFromSmiles successfully parsed it
            valid_smiles.append(smi)
            qed_scores.append(QED.qed(mol))
            mw_values.append(Descriptors.MolWt(mol))
            logp_values.append(Crippen.MolLogP(mol))
    except:
        continue

print(f"✅ Valid molecules: {len(valid_smiles)} / {len(generated_smiles)}")
print(f"📊 Validity %: {(len(valid_smiles)/len(generated_smiles))*100:.2f}%")

# ========================
# PRINT SAMPLE SMILES
# ========================
print("\n🧾 SAMPLE VALID SMILES (Top 10):")
print("-"*60)

for i, smi in enumerate(valid_smiles[:10]):
    print(f"{i+1:02d}: {smi}")

# ========================
# METRICS
# ========================
if len(qed_scores) > 0:
    print("\n📊 MOLECULAR PROPERTY STATISTICS")
    print("-"*60)
    print(f"Avg QED Score     : {np.mean(qed_scores):.3f}")
    print(f"Avg Mol Weight    : {np.mean(mw_values):.2f}")
    print(f"Avg LogP          : {np.mean(logp_values):.2f}")

# ========================
# PLOTS
# ========================
print("\n📈 Generating plots...")

plt.figure()
plt.hist(qed_scores, bins=20)
plt.title("QED Distribution")
plt.xlabel("QED")
plt.ylabel("Count")
plt.show()
plt.close()

plt.figure()
plt.hist(mw_values, bins=20)
plt.title("Molecular Weight Distribution")
plt.xlabel("MW")
plt.ylabel("Count")
plt.show()
plt.close()

plt.figure()
plt.hist(logp_values, bins=20)
plt.title("LogP Distribution")
plt.xlabel("LogP")
plt.ylabel("Count")
plt.show()
plt.close()

# ========================
# MOLECULAR VISUALIZATION
# ========================
print("\n🧬 Visualizing Top Molecules (by QED)...")

try:
    # Sort by QED
    sorted_data = sorted(
        zip(valid_smiles, qed_scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_smiles = [s for s, _ in sorted_data[:10]]
    top_mols = [Chem.MolFromSmiles(s) for s in top_smiles]

    img = Draw.MolsToGridImage(
        top_mols,
        molsPerRow=5,
        subImgSize=(200,200),
        legends=[f"QED:{q:.2f}" for _, q in sorted_data[:10]]
    )

    display(img)

    # ========================
    # SAVE RESULTS
    # ========================
    print("\n💾 Saving results to Google Drive...")

    df = pd.DataFrame({
        "SMILES": valid_smiles,
        "QED": qed_scores,
        "MW": mw_values,
        "LogP": logp_values
    })

    # Ensure the directory exists before saving
    os.makedirs(f"{SAVE_DIR}/results", exist_ok=True)

    csv_path = f"{SAVE_DIR}/results/generated_results.csv"
    img_path = f"{SAVE_DIR}/results/top_molecules.png"

    df.to_csv(csv_path, index=False)

    # Convert image to bytes, then to PIL Image, then to NumPy array, and save.
    # This is more robust against potential non-standard 'Image' objects returned by RDKit.
    png_bytes = img._repr_png_() # Get PNG representation from the object

    if png_bytes:
        pil_image = PILImage.open(BytesIO(png_bytes))
        plt.imsave(img_path, np.array(pil_image))
    else:
        # Fallback to the original method if _repr_png_() fails or returns None
        print("Warning: Could not get PNG representation from image object, attempting direct numpy conversion.")
        plt.imsave(img_path, np.array(img)) # This was the line that previously failed

    print(f"✅ CSV saved at: {csv_path}")
    print(f"✅ Image saved at: {img_path}")

except Exception as e:
    print(f"❌ An error occurred during molecular visualization or saving: {e}")

print("\n" + "="*60)
print("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
print("="*60)
# =========================================================
# DRUG-LIKENESS FILTER + PRINTING BLOCK
# =========================================================

from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Crippen, Lipinski
import pandas as pd
import numpy as np

print("\n🧪 Applying Drug-Likeness Filtering (Lipinski Rule)...")

results = []

for smi in gen_smiles:
    try:
        mol = Chem.MolFromSmiles(smi)

        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            qed = QED.qed(mol)

            # ✅ Lipinski Rule
            if (mw < 500 and -2 < logp < 5 and hbd <= 5 and hba <= 10):

                results.append({
                    "SMILES": smi,
                    "QED": qed,
                    "MW": mw,
                    "LogP": logp,
                    "HBD": hbd,
                    "HBA": hba
                })

    except:
        continue

df_filtered = pd.DataFrame(results)

# =========================================================
# 📊 FINAL METRICS
# =========================================================
total_generated = len(gen_smiles)
valid_count = len([s for s in gen_smiles if Chem.MolFromSmiles(s)])
drug_like_count = len(df_filtered)
unique_count = len(set(gen_smiles))

print("\n📊 FINAL EVALUATION METRICS")
print("="*50)

print(f"Total Generated        : {total_generated}")
print(f"Valid Molecules        : {valid_count}")
print(f"Drug-like Molecules    : {drug_like_count}")
print(f"Validity %             : {(valid_count/total_generated)*100:.2f}%")
print(f"Drug-like %            : {(drug_like_count/total_generated)*100:.2f}%")
print(f"Unique Molecules       : {unique_count}")

if drug_like_count > 0:
    print(f"\nAvg QED               : {df_filtered['QED'].mean():.3f}")
    print(f"Avg Molecular Weight  : {df_filtered['MW'].mean():.2f}")
    print(f"Avg LogP              : {df_filtered['LogP'].mean():.2f}")

# =========================================================
# 🧾 PRINT SAMPLE SMILES
# =========================================================
print("\n🧾 TOP DRUG-LIKE MOLECULES (Top 10 by QED)")
print("-"*60)

top_df = df_filtered.sort_values(by="QED", ascending=False).head(10)

for i, row in top_df.iterrows():
    print(f"{i+1:02d}: {row['SMILES']}")
    print(f"     QED={row['QED']:.3f} | MW={row['MW']:.1f} | LogP={row['LogP']:.2f} | HBD={row['HBD']} | HBA={row['HBA']}")
    print()

# =========================================================
# 💾 SAVE RESULTS (OPTIONAL BUT RECOMMENDED)
# =========================================================
save_path = f"{SAVE_DIR}/drug_like_molecules.csv"
df_filtered.to_csv(save_path, index=False)

print(f"✅ Drug-like molecules saved to: {save_path}")
