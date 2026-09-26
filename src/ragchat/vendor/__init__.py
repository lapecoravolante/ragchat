"""Package vendor: librerie e componenti bundled con l'applicazione.

All'importazione di questo package vengono automaticamente registrate le
DLL Windows necessarie (VC++ runtime e llama_cpp native), in modo che
qualsiasi modulo che importa ``ragchat.vendor`` abbia gia' le DLL
disponibili nel loader di sistema prima di caricare llama_cpp.
"""

from ragchat.vendor.win_dll_loader import register_windows_dll_dirs

# Registra le DLL all'importazione del package (idempotente su non-Windows)
register_windows_dll_dirs()
