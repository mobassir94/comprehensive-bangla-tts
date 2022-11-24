import sys
import os
import gdown
import pandas as pd


import soundfile as sf
import shutil
import bangla


try:
  from TTS.utils.synthesizer import Synthesizer
except:
  print("coundn't import TTS synthesizer,trying again!")
#from TTS.utils.synthesizer import Synthesizer

import TTS

# from https://github.com/coqui-ai/TTS/blob/dev/TTS/utils/synthesizer.py
import time
from typing import List

import numpy as np
import pysbd
import torch

from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model

# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
from TTS.tts.utils.synthesis import synthesis, transfer_voice, trim_silence
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.models import setup_model as setup_vocoder_model
from TTS.vocoder.utils.generic_utils import interpolate_vocoder_input
sys.path.append('./klaam')
from klaam import TextToSpeech
from pyarabic.araby import strip_diacritics

male = True

root_path = "./klaam/"
prepare_tts_model_path = "./klaam/cfgs/FastSpeech2/config/Arabic/preprocess.yaml"
model_config_path = "./klaam/cfgs/FastSpeech2/config/Arabic/model.yaml"
train_config_path = "./klaam/cfgs/FastSpeech2/config/Arabic/train.yaml"
vocoder_config_path = "./klaam/cfgs/FastSpeech2/model_config/hifigan/config.json"
speaker_pre_trained_path = "./klaam/data/model_weights/hifigan/generator_universal.pth.tar"

ar_model = TextToSpeech(prepare_tts_model_path, model_config_path, train_config_path, vocoder_config_path, speaker_pre_trained_path,root_path)


class Synthesizer(object):
    def __init__(
        self,
        tts_checkpoint: str,
        tts_config_path: str,
        tts_speakers_file: str = "",
        tts_languages_file: str = "",
        vocoder_checkpoint: str = "",
        vocoder_config: str = "",
        encoder_checkpoint: str = "",
        encoder_config: str = "",
        use_cuda: bool = False,
    ) -> None:
        """General 🐸 TTS interface for inference. It takes a tts and a vocoder
        model and synthesize speech from the provided text.
        The text is divided into a list of sentences using `pysbd` and synthesize
        speech on each sentence separately.
        If you have certain special characters in your text, you need to handle
        them before providing the text to Synthesizer.
        TODO: set the segmenter based on the source language
        Args:
            tts_checkpoint (str): path to the tts model file.
            tts_config_path (str): path to the tts config file.
            vocoder_checkpoint (str, optional): path to the vocoder model file. Defaults to None.
            vocoder_config (str, optional): path to the vocoder config file. Defaults to None.
            encoder_checkpoint (str, optional): path to the speaker encoder model file. Defaults to `""`,
            encoder_config (str, optional): path to the speaker encoder config file. Defaults to `""`,
            use_cuda (bool, optional): enable/disable cuda. Defaults to False.
        """
        self.tts_checkpoint = tts_checkpoint
        self.tts_config_path = tts_config_path
        self.tts_speakers_file = tts_speakers_file
        self.tts_languages_file = tts_languages_file
        self.vocoder_checkpoint = vocoder_checkpoint
        self.vocoder_config = vocoder_config
        self.encoder_checkpoint = encoder_checkpoint
        self.encoder_config = encoder_config
        self.use_cuda = use_cuda

        self.tts_model = None
        self.vocoder_model = None
        self.speaker_manager = None
        self.num_speakers = 0
        self.tts_speakers = {}
        self.language_manager = None
        self.num_languages = 0
        self.tts_languages = {}
        self.d_vector_dim = 0
        self.seg = self._get_segmenter("en")
        self.use_cuda = use_cuda

        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."
        self._load_tts(tts_checkpoint, tts_config_path, use_cuda)
        self.output_sample_rate = self.tts_config.audio["sample_rate"]
        if vocoder_checkpoint:
            self._load_vocoder(vocoder_checkpoint, vocoder_config, use_cuda)
            self.output_sample_rate = self.vocoder_config.audio["sample_rate"]

    @staticmethod
    def _get_segmenter(lang: str):
        """get the sentence segmenter for the given language.
        Args:
            lang (str): target language code.
        Returns:
            [type]: [description]
        """
        return pysbd.Segmenter(language=lang, clean=True)

    def _load_tts(self, tts_checkpoint: str, tts_config_path: str, use_cuda: bool) -> None:
        """Load the TTS model.
        1. Load the model config.
        2. Init the model from the config.
        3. Load the model weights.
        4. Move the model to the GPU if CUDA is enabled.
        5. Init the speaker manager in the model.
        Args:
            tts_checkpoint (str): path to the model checkpoint.
            tts_config_path (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        # pylint: disable=global-statement
        self.tts_config = load_config(tts_config_path)
        if self.tts_config["use_phonemes"] and self.tts_config["phonemizer"] is None:
            raise ValueError("Phonemizer is not defined in the TTS config.")

        self.tts_model = setup_tts_model(config=self.tts_config)

        if not self.encoder_checkpoint:
            self._set_speaker_encoder_paths_from_tts_config()

        self.tts_model.load_checkpoint(self.tts_config, tts_checkpoint, eval=True)
        if use_cuda:
            self.tts_model.cuda()

        if self.encoder_checkpoint and hasattr(self.tts_model, "speaker_manager"):
            self.tts_model.speaker_manager.init_encoder(self.encoder_checkpoint, self.encoder_config, use_cuda)

    def _set_speaker_encoder_paths_from_tts_config(self):
        """Set the encoder paths from the tts model config for models with speaker encoders."""
        if hasattr(self.tts_config, "model_args") and hasattr(
            self.tts_config.model_args, "speaker_encoder_config_path"
        ):
            self.encoder_checkpoint = self.tts_config.model_args.speaker_encoder_model_path
            self.encoder_config = self.tts_config.model_args.speaker_encoder_config_path

    def _load_vocoder(self, model_file: str, model_config: str, use_cuda: bool) -> None:
        """Load the vocoder model.
        1. Load the vocoder config.
        2. Init the AudioProcessor for the vocoder.
        3. Init the vocoder model from the config.
        4. Move the model to the GPU if CUDA is enabled.
        Args:
            model_file (str): path to the model checkpoint.
            model_config (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        self.vocoder_config = load_config(model_config)
        self.vocoder_ap = AudioProcessor(verbose=False, **self.vocoder_config.audio)
        self.vocoder_model = setup_vocoder_model(self.vocoder_config)
        self.vocoder_model.load_checkpoint(self.vocoder_config, model_file, eval=True)
        if use_cuda:
            self.vocoder_model.cuda()

    def split_into_sentences(self, text) -> List[str]:
        """Split give text into sentences.
        Args:
            text (str): input text in string format.
        Returns:
            List[str]: list of sentences.
        """
        return self.seg.segment(text)

    def save_wav(self, wav: List[int], path: str) -> None:
        """Save the waveform as a file.
        Args:
            wav (List[int]): waveform as a list of values.
            path (str): output path to save the waveform.
        """
        wav = np.array(wav)
        self.tts_model.ap.save_wav(wav, path, self.output_sample_rate)

    def tts(
        self,
        text: str = "",
        speaker_name: str = "",
        language_name: str = "",
        speaker_wav=None,
        style_wav=None,
        style_text=None,
        reference_wav=None,
        reference_speaker_name=None,
    ) -> List[int]:
        """🐸 TTS magic. Run all the models and generate speech.
        Args:
            text (str): input text.
            speaker_name (str, optional): spekaer id for multi-speaker models. Defaults to "".
            language_name (str, optional): language id for multi-language models. Defaults to "".
            speaker_wav (Union[str, List[str]], optional): path to the speaker wav. Defaults to None.
            style_wav ([type], optional): style waveform for GST. Defaults to None.
            style_text ([type], optional): transcription of style_wav for Capacitron. Defaults to None.
            reference_wav ([type], optional): reference waveform for voice conversion. Defaults to None.
            reference_speaker_name ([type], optional): spekaer id of reference waveform. Defaults to None.
        Returns:
            List[int]: [description]
        """
        start_time = time.time()
        wavs = []

        if not text and not reference_wav:
            raise ValueError(
                "You need to define either `text` (for sythesis) or a `reference_wav` (for voice conversion) to use the Coqui TTS API."
            )

        if text:
            sens = self.split_into_sentences(text)
            # print(" > Text splitted to sentences.")
            # print(sens)

        # handle multi-speaker
        speaker_embedding = None
        speaker_id = None
        if self.tts_speakers_file or hasattr(self.tts_model.speaker_manager, "name_to_id"):
            if speaker_name and isinstance(speaker_name, str):
                if self.tts_config.use_d_vector_file:
                    # get the average speaker embedding from the saved d_vectors.
                    speaker_embedding = self.tts_model.speaker_manager.get_mean_embedding(
                        speaker_name, num_samples=None, randomize=False
                    )
                    speaker_embedding = np.array(speaker_embedding)[None, :]  # [1 x embedding_dim]
                else:
                    # get speaker idx from the speaker name
                    speaker_id = self.tts_model.speaker_manager.name_to_id[speaker_name]

            elif not speaker_name and not speaker_wav:
                raise ValueError(
                    " [!] Look like you use a multi-speaker model. "
                    "You need to define either a `speaker_name` or a `speaker_wav` to use a multi-speaker model."
                )
            else:
                speaker_embedding = None
        else:
            if speaker_name:
                raise ValueError(
                    f" [!] Missing speakers.json file path for selecting speaker {speaker_name}."
                    "Define path for speaker.json if it is a multi-speaker model or remove defined speaker idx. "
                )

        # handle multi-lingaul
        language_id = None
        if self.tts_languages_file or (
            hasattr(self.tts_model, "language_manager") and self.tts_model.language_manager is not None
        ):
            if language_name and isinstance(language_name, str):
                language_id = self.tts_model.language_manager.name_to_id[language_name]

            elif not language_name:
                raise ValueError(
                    " [!] Look like you use a multi-lingual model. "
                    "You need to define either a `language_name` or a `style_wav` to use a multi-lingual model."
                )

            else:
                raise ValueError(
                    f" [!] Missing language_ids.json file path for selecting language {language_name}."
                    "Define path for language_ids.json if it is a multi-lingual model or remove defined language idx. "
                )

        # compute a new d_vector from the given clip.
        if speaker_wav is not None:
            speaker_embedding = self.tts_model.speaker_manager.compute_embedding_from_clip(speaker_wav)

        use_gl = self.vocoder_model is None

        if not reference_wav:
            for sen in sens:
                # synthesize voice
                outputs = synthesis(
                    model=self.tts_model,
                    text=sen,
                    CONFIG=self.tts_config,
                    use_cuda=self.use_cuda,
                    speaker_id=speaker_id,
                    style_wav=style_wav,
                    style_text=style_text,
                    use_griffin_lim=use_gl,
                    d_vector=speaker_embedding,
                    language_id=language_id,
                )
                waveform = outputs["wav"]
                mel_postnet_spec = outputs["outputs"]["model_outputs"][0].detach().cpu().numpy()
                if not use_gl:
                    # denormalize tts output based on tts audio config
                    mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
                    device_type = "cuda" if self.use_cuda else "cpu"
                    # renormalize spectrogram based on vocoder config
                    vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                    # compute scale factor for possible sample rate mismatch
                    scale_factor = [
                        1,
                        self.vocoder_config["audio"]["sample_rate"] / self.tts_model.ap.sample_rate,
                    ]
                    if scale_factor[1] != 1:
                        print(" > interpolating tts model output.")
                        vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                    else:
                        vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                    # run vocoder model
                    # [1, T, C]
                    waveform = self.vocoder_model.inference(vocoder_input.to(device_type))
                if self.use_cuda and not use_gl:
                    waveform = waveform.cpu()
                if not use_gl:
                    waveform = waveform.numpy()
                waveform = waveform.squeeze()

                # trim silence
                if "do_trim_silence" in self.tts_config.audio and self.tts_config.audio["do_trim_silence"]:
                    waveform = trim_silence(waveform, self.tts_model.ap)

                wavs += list(waveform)
                wavs += [0] * 10000
        else:
            # get the speaker embedding or speaker id for the reference wav file
            reference_speaker_embedding = None
            reference_speaker_id = None
            if self.tts_speakers_file or hasattr(self.tts_model.speaker_manager, "name_to_id"):
                if reference_speaker_name and isinstance(reference_speaker_name, str):
                    if self.tts_config.use_d_vector_file:
                        # get the speaker embedding from the saved d_vectors.
                        reference_speaker_embedding = self.tts_model.speaker_manager.get_embeddings_by_name(
                            reference_speaker_name
                        )[0]
                        reference_speaker_embedding = np.array(reference_speaker_embedding)[
                            None, :
                        ]  # [1 x embedding_dim]
                    else:
                        # get speaker idx from the speaker name
                        reference_speaker_id = self.tts_model.speaker_manager.name_to_id[reference_speaker_name]
                else:
                    reference_speaker_embedding = self.tts_model.speaker_manager.compute_embedding_from_clip(
                        reference_wav
                    )
            outputs = transfer_voice(
                model=self.tts_model,
                CONFIG=self.tts_config,
                use_cuda=self.use_cuda,
                reference_wav=reference_wav,
                speaker_id=speaker_id,
                d_vector=speaker_embedding,
                use_griffin_lim=use_gl,
                reference_speaker_id=reference_speaker_id,
                reference_d_vector=reference_speaker_embedding,
            )
            waveform = outputs
            if not use_gl:
                mel_postnet_spec = outputs[0].detach().cpu().numpy()
                # denormalize tts output based on tts audio config
                mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
                device_type = "cuda" if self.use_cuda else "cpu"
                # renormalize spectrogram based on vocoder config
                vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                # compute scale factor for possible sample rate mismatch
                scale_factor = [
                    1,
                    self.vocoder_config["audio"]["sample_rate"] / self.tts_model.ap.sample_rate,
                ]
                if scale_factor[1] != 1:
                    print(" > interpolating tts model output.")
                    vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                else:
                    vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                # run vocoder model
                # [1, T, C]
                waveform = self.vocoder_model.inference(vocoder_input.to(device_type))
            if self.use_cuda:
                waveform = waveform.cpu()
            if not use_gl:
                waveform = waveform.numpy()
            wavs = waveform.squeeze()

        # compute stats
        process_time = time.time() - start_time
        audio_time = len(wavs) / self.tts_config.audio["sample_rate"]
        # print(f" > Processing time: {process_time}")
        # print(f" > Real-time factor: {process_time / audio_time}")
        return wavs



# from https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/utils/text/tokenizer.py

from typing import Callable, Dict, List, Union

from TTS.tts.utils.text import cleaners
from TTS.tts.utils.text.characters import Graphemes, IPAPhonemes
from TTS.tts.utils.text.phonemizers import DEF_LANG_TO_PHONEMIZER, get_phonemizer_by_name
from TTS.utils.generic_utils import get_import_path, import_class


class TTSTokenizer:
    """🐸TTS tokenizer to convert input characters to token IDs and back.
    Token IDs for OOV chars are discarded but those are stored in `self.not_found_characters` for later.
    Args:
        use_phonemes (bool):
            Whether to use phonemes instead of characters. Defaults to False.
        characters (Characters):
            A Characters object to use for character-to-ID and ID-to-character mappings.
        text_cleaner (callable):
            A function to pre-process the text before tokenization and phonemization. Defaults to None.
        phonemizer (Phonemizer):
            A phonemizer object or a dict that maps language codes to phonemizer objects. Defaults to None.
    Example:
        >>> from TTS.tts.utils.text.tokenizer import TTSTokenizer
        >>> tokenizer = TTSTokenizer(use_phonemes=False, characters=Graphemes())
        >>> text = "Hello world!"
        >>> ids = tokenizer.text_to_ids(text)
        >>> text_hat = tokenizer.ids_to_text(ids)
        >>> assert text == text_hat
    """

    def __init__(
        self,
        use_phonemes=False,
        text_cleaner: Callable = None,
        characters: "BaseCharacters" = None,
        phonemizer: Union["Phonemizer", Dict] = None,
        add_blank: bool = False,
        use_eos_bos=False,
    ):
        self.text_cleaner = text_cleaner
        self.use_phonemes = use_phonemes
        self.add_blank = add_blank
        self.use_eos_bos = use_eos_bos
        self.characters = characters
        self.not_found_characters = []
        self.phonemizer = phonemizer

    @property
    def characters(self):
        return self._characters

    @characters.setter
    def characters(self, new_characters):
        self._characters = new_characters
        self.pad_id = self.characters.char_to_id(self.characters.pad) if self.characters.pad else None
        self.blank_id = self.characters.char_to_id(self.characters.blank) if self.characters.blank else None

    def encode(self, text: str) -> List[int]:
        """Encodes a string of text as a sequence of IDs."""
        token_ids = []
        for char in text:
            try:
                idx = self.characters.char_to_id(char)
                token_ids.append(idx)
            except KeyError:
                # discard but store not found characters
                if char not in self.not_found_characters:
                    self.not_found_characters.append(char)
                    # print(text)
                    # print(f" [!] Character {repr(char)} not found in the vocabulary. Discarding it.")
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a sequence of IDs to a string of text."""
        text = ""
        for token_id in token_ids:
            text += self.characters.id_to_char(token_id)
        return text

    def text_to_ids(self, text: str, language: str = None) -> List[int]:  # pylint: disable=unused-argument
        """Converts a string of text to a sequence of token IDs.
        Args:
            text(str):
                The text to convert to token IDs.
            language(str):
                The language code of the text. Defaults to None.
        TODO:
            - Add support for language-specific processing.
        1. Text normalizatin
        2. Phonemization (if use_phonemes is True)
        3. Add blank char between characters
        4. Add BOS and EOS characters
        5. Text to token IDs
        """
        # TODO: text cleaner should pick the right routine based on the language
        if self.text_cleaner is not None:
            text = self.text_cleaner(text)
        if self.use_phonemes:
            text = self.phonemizer.phonemize(text, separator="")
        if self.add_blank:
            text = self.intersperse_blank_char(text, True)
        if self.use_eos_bos:
            text = self.pad_with_bos_eos(text)
        return self.encode(text)

    def ids_to_text(self, id_sequence: List[int]) -> str:
        """Converts a sequence of token IDs to a string of text."""
        return self.decode(id_sequence)

    def pad_with_bos_eos(self, char_sequence: List[str]):
        """Pads a sequence with the special BOS and EOS characters."""
        return [self.characters.bos] + list(char_sequence) + [self.characters.eos]

    def intersperse_blank_char(self, char_sequence: List[str], use_blank_char: bool = False):
        """Intersperses the blank character between characters in a sequence.
        Use the ```blank``` character if defined else use the ```pad``` character.
        """
        char_to_use = self.characters.blank if use_blank_char else self.characters.pad
        result = [char_to_use] * (len(char_sequence) * 2 + 1)
        result[1::2] = char_sequence
        return result

    def print_logs(self, level: int = 0):
        indent = "\t" * level
        print(f"{indent}| > add_blank: {self.add_blank}")
        print(f"{indent}| > use_eos_bos: {self.use_eos_bos}")
        print(f"{indent}| > use_phonemes: {self.use_phonemes}")
        if self.use_phonemes:
            print(f"{indent}| > phonemizer:")
            self.phonemizer.print_logs(level + 1)
        if len(self.not_found_characters) > 0:
            print(f"{indent}| > {len(self.not_found_characters)} not found characters:")
            for char in self.not_found_characters:
                print(f"{indent}| > {char}")

    @staticmethod
    def init_from_config(config: "Coqpit", characters: "BaseCharacters" = None):
        """Init Tokenizer object from config
        Args:
            config (Coqpit): Coqpit model config.
            characters (BaseCharacters): Defines the model character set. If not set, use the default options based on
                the config values. Defaults to None.
        """
        # init cleaners
        text_cleaner = None
        if isinstance(config.text_cleaner, (str, list)):
            text_cleaner = getattr(cleaners, config.text_cleaner)

        # init characters
        if characters is None:
            # set characters based on defined characters class
            if config.characters and config.characters.characters_class:
                CharactersClass = import_class(config.characters.characters_class)
                characters, new_config = CharactersClass.init_from_config(config)
            # set characters based on config
            else:
                if config.use_phonemes:
                    # init phoneme set
                    characters, new_config = IPAPhonemes().init_from_config(config)
                else:
                    # init character set
                    characters, new_config = Graphemes().init_from_config(config)

        else:
            characters, new_config = characters.init_from_config(config)

        # set characters class
        new_config.characters.characters_class = get_import_path(characters)

        # init phonemizer
        phonemizer = None
        if config.use_phonemes:
            phonemizer_kwargs = {"language": config.phoneme_language}

            if "phonemizer" in config and config.phonemizer:
                phonemizer = get_phonemizer_by_name(config.phonemizer, **phonemizer_kwargs)
            else:
                try:
                    phonemizer = get_phonemizer_by_name(
                        DEF_LANG_TO_PHONEMIZER[config.phoneme_language], **phonemizer_kwargs
                    )
                    new_config.phonemizer = phonemizer.name()
                except KeyError as e:
                    raise ValueError(
                        f"""No phonemizer found for language {config.phoneme_language}.
                        You may need to install a third party library for this language."""
                    ) from e

        return (
            TTSTokenizer(
                config.use_phonemes, text_cleaner, characters, phonemizer, config.add_blank, config.enable_eos_bos_chars
            ),
            new_config,
        )
    
# link -> hhttps://drive.google.com/drive/folders/1IMCiQpyYBqu98dlRMSINjFNc34fI6zhs?usp=sharing
url = "https://drive.google.com/drive/folders/1IMCiQpyYBqu98dlRMSINjFNc34fI6zhs?usp=sharing"
isExist = os.path.exists('./bangla_tts')
if not isExist:
    gdown.download_folder(url=url, quiet=True, use_cookies=False)   


if(male):
  # test_ckpt = 'bangla_tts/bn_glow_tts/male/checkpoint_328000.pth'
  # test_config = 'bangla_tts/bn_glow_tts/male/config.json'

  test_ckpt = 'bangla_tts/bn_vits/male/checkpoint_910000.pth'
  test_config = 'bangla_tts/bn_vits/male/config.json'

else:
  test_ckpt = 'bangla_tts/bn_glow_tts/female/checkpoint_180000.pth'
  test_config = 'bangla_tts/bn_glow_tts/female/config.json'

bn_model=Synthesizer(test_ckpt,test_config)

import re
import torchaudio.functional as F
import torchaudio
from bnnumerizer import numerize
import gc
from bnunicodenormalizer import Normalizer 
from pydub import AudioSegment
from pyarabic.araby import strip_diacritics

# initialize
bnorm=Normalizer()

# Create empty audio file of half second duration (purpose -> post processing)

audio = AudioSegment.silent(duration=500)
sound = audio.set_frame_rate(audio.frame_rate*2)
sound.export("./empty.wav", format="wav")

#for worst case scenario
audio = AudioSegment.silent(duration=50)
sound = audio.set_frame_rate(audio.frame_rate*2)
sound.export("./empty_chunk.wav", format="wav")
empty_audio_chunk, rate_of_sample = torchaudio.load('./empty_chunk.wav')
empty_audio_chunk = empty_audio_chunk.flatten()


#loading empty audio file of  1 second to append before and after each arabic chunk for increasing mlt reading rhythm.

empty_audio, rate_of_sample = torchaudio.load('./empty.wav')
empty_audio = empty_audio.flatten()


def normalize(sen):
    _words = [bnorm(word)['normalized']  for word in sen.split()]
    return " ".join([word for word in _words if word is not None]) 


class BigTextToAudio(object):
    
    def __init__(self,
                 ar_model = ar_model,
                 bn_model = bn_model,
                 ar_sample_rate=22050,
                 bn_sample_rate=22050,
                 out_sample_rate=22050,
                 
                 find_nd_replace={
                                  "কেন"  : "কেনো",
                                  "কোন" : "কোনো",
                                  "বল"   : "বলো",
                                  "চল"   : "চলো",
                                  "কর"   : "করো",
                                  "রাখ"   : "রাখো",
                                     },
                 
                 attribution_dict={"সাঃ":"সাল্লাল্লাহু আলাইহি ওয়া সাল্লাম",    
                                   "স.":"সাল্লাল্লাহু আলাইহি ওয়া সাল্লাম",                
                                  "আঃ":"আলাইহিস সালাম",
                                  "রাঃ":"রাদিআল্লাহু আনহু",
                                  "রহঃ":"রহমাতুল্লাহি আলাইহি",
                                  "রহিঃ":"রহিমাহুল্লাহ",
                                  "হাফিঃ":"হাফিযাহুল্লাহ",
                                  "দাঃবাঃ":"দামাত বারাকাতুহুম,দামাত বারাকাতুল্লাহ",
                                   "/"   : " বাই ",
                                  },
                resample_params={"lowpass_filter_width": 64,
                                "rolloff": 0.9475937167399596,
                                "resampling_method": "kaiser_window",
                                "beta": 14.769656459379492}
                ):
        '''
            Instantiates a Big Text to Audio conversion object for bangla and arabic
            args:
                ar_model : arabic tts model
                bn_model : bangla tts model
                ar_sample_rate : arabic audio sample rate [optional] default: 22050
                bn_sample_rate : bangla audio sample rate [optional] default: 22050
                out_sample_rate : audio sample rate [optional] default: 22050
                attribution_dict : a dict for attribution expansion [optional]
                resample_params : audio resampling parameters [optional]
            resources:
                # Main class: modified from https://github.com/snakers4/silero-models/pull/174
                # Audio converter:https://www.kaggle.com/code/shahruk10/inference-notebook-wav2vec2
        '''
        self.ar_model = ar_model
        self.bn_model = bn_model

        self.find_nd_replace=find_nd_replace
        self.attribution_dict=attribution_dict
        self.ar_sample_rate=ar_sample_rate
        self.bn_sample_rate=bn_sample_rate
        self.sample_rate=out_sample_rate  
        self.resample_params=resample_params
    
    #https://gist.github.com/mohabmes/33b724edfd4f0f3ec2e6644168db516e
    def removeUnnecessarySpaces(self,text):
        return re.sub(r'[\n\t\ ]+', ' ', text)
    def removeNonArabicChar(self,text):
        return re.sub(r'[^0-9\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD.0-9]+', ' ', text)

    # public
    def ar_tts(self,text):
        '''
            args: 
                text: arabic text (string)
            returns:
                audio as torch tensor
        '''
        #text = text.replace('،','')
        text = re.sub('،', '', text) # replace any ، with '' 
        text = re.sub('٭', '', text) # replace any ٭ with '' 
        text = strip_diacritics(text)
        ar_text = self.removeUnnecessarySpaces(text)
        ar_text = self.removeNonArabicChar(ar_text)
        ar_text = self.removeUnnecessarySpaces(ar_text)


        try:
            self.ar_model.synthesize(ar_text.strip())
            audio, rate_of_sample = torchaudio.load('./sample.wav')
            audio = audio.flatten()
            audio = torch.cat([empty_audio,audio], axis=0) #start empty
            audio = torch.cat([audio,empty_audio], axis=0) #end empty
           
        except:
            print("--------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>    failed ar =>",text,"end")
            text = re.sub(u'[^\u0980-\u09FF]+', ' ', text) # get bangla only
            if not text.strip():
                audio = empty_audio_chunk
            else:
                audio=self.bn_tts(text)
  
                  
        
        return audio
    # public
    def bn_tts(self,text):
        '''
            args: 
                text   : bangla text (string)
            returns:
                audio as torch tensor
        '''
        return torch.as_tensor(self.bn_model.tts(text))
    
    # public
    def expand_full_attribution(self,text):
        for word in self.attribution_dict:
            if word in text:
                text = text.replace(word, normalize(self.attribution_dict[word]))
        return text
    
    def exact_replacement(self,text):
        for word,replacement in self.find_nd_replace.items():
            text = re.sub(normalize(word),normalize(replacement),text)
        return text

    def collapse_whitespace(self,text):
        # Regular expression matching whitespace:
        _whitespace_re = re.compile(r"\s+")
        return re.sub(_whitespace_re, " ", text)

    # public
    def tag_text(self,text):
        '''
            * tags arabic text with <ar>text</ar>
            * tags bangla text with <bn>text</bn>
        '''
        # remove multiple spaces
        text=re.sub(' +', ' ',text)
        # create start and end
        text="start"+text+"end"
        # tag text
        parts=re.split(u'[\u0600-\u06FF]+', text)
        # remove non chars
        parts=[p for p in parts if p.strip()]
        # unique parts
        parts=set(parts)
        # tag the text
        for m in parts:
            if len(m.strip())>1:text=text.replace(m,f"</ar><SPLIT><bn>{m}</bn><SPLIT><ar>")
        # clean-tags
        text=text.replace("</ar><SPLIT><bn>start",'<bn>')
        text=text.replace("end</bn><SPLIT><ar>",'</bn>')
        return text

    def process_text(self,text):
        '''
        process multilingual text for suitable MLT TTS format
            * expand attributions
            * numerize text
            * tag sections of the text
            * sequentially list text blocks
            * Split based on sentence ending Characters

        '''
        
        # english numbers to bangla conversion
        res = re.search('[0-9]', text)
        if res is not None:
          text = bangla.convert_english_digit_to_bangla_digit(text)
        
        #replace ':' in between two bangla numbers with ' এর '
        pattern=r"[০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯]:[০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯]"
        matches=re.findall(pattern,text)
        for m in matches:
            r=m.replace(":"," এর ")
            text=text.replace(m,r)

        # numerize text
        try:
          text=numerize(text)
        except:
          print("couldn't numerize bengali.")
        # tag sections
        # text=self.tag_text(text)

        #text="।".join([self.tag_text(line) for line in text.split("।")])
        
        if "।" in text:punct="।"
        elif "." in text:punct="."
        else:punct="\n"

        text=punct.join([self.tag_text(line) for line in text.split(punct)])

        # text blocks
        blocks=text.split("<SPLIT>")
        blocks=[b for b in blocks if b.strip()]
        # create tuple of (lang,text)
        data=[]
        for block in blocks:
            lang=None
            if "<bn>" in block:
                block=block.replace("<bn>",'').replace("</bn>",'')
                lang="bn"
            elif "<ar>" in block:
                block=block.replace("<ar>",'').replace("</ar>",'')
                lang="ar"
            
            # Split based on sentence ending Characters

            if lang == "bn":
              bn_text = block.strip()

              sentenceEnders = re.compile('[।!?]')
              sentences = sentenceEnders.split(str(bn_text))

              for i in range(len(sentences)):
                  res = re.sub('\n','',sentences[i])
                  res = normalize(res)
                  # expand attributes
                  res=self.expand_full_attribution(res)
                  
                  #res=self.exact_replacement(res)
                    
                  res = self.collapse_whitespace(res)
#                   res += '।'
                  if(len(res)>500):
                      firstpart, secondpart = res[:len(res)//2], res[len(res)//2:]
                      data.append((lang,firstpart))
                      data.append((lang,secondpart))
                  else:
                      data.append((lang,res))

            elif lang == "ar":
                ar_text = block.strip()
                ar_text = re.sub("؟", "?", ar_text) # replace any ؟ with ?

                sentenceEnders = re.compile('[.،!?]')
                sentences = sentenceEnders.split(str(ar_text))

                for i in range(len(sentences)):
                    res = re.sub('\n','',sentences[i])
                    res = self.collapse_whitespace(res)
                    
                    if(len(res)>500):
                      firstpart, secondpart = res[:len(res)//2], res[len(res)//2:]
                      data.append((lang,firstpart))
                      data.append((lang,secondpart))
                    else:
                        data.append((lang,res))

                    #data.append((lang,res))
                    
        return data
    
    def resample_audio(self,audio,sr):
        '''
            resample audio with sample rate
            args:
                audio : torch.tensor audio
                sr: audi sample rate
        '''
        if sr==self.sample_rate:
            return audio
        else:
            return F.resample(audio,sr,self.sample_rate,**self.resample_params)
        
    
    def get_audio(self,data):
        '''
            creates audio from given data 
                * data=List[Tuples(lang,text)]
        '''
        audio_list = []
        for block in data:
            lang,text=block
            if not text.strip():
                continue
            if lang=="bn":
                audio=self.bn_tts(text)
                sr=self.bn_sample_rate
            else:
                audio=self.ar_tts(text)
                sr=self.ar_sample_rate
            
            if self.resample_audio_to_out_sample_rate:
                audio=self.resample_audio(audio,sr)
                
            audio_list.append(audio)
  
        audio = torch.cat([k for k in audio_list])
        return audio
    
    # call
    def __call__(self,text,resample_audio_to_out_sample_rate=True):
        '''
            args: 
                text   : bangla text (string)
                resample_audio_to_out_sample_rate: for different sample rate in different models, resample the output audio 
                                                   in uniform sample rate 
                                                   * default:True
            returns:
                audio as numpy data
        '''
        self.resample_audio_to_out_sample_rate=resample_audio_to_out_sample_rate
        data=self.process_text(text)
        audio=self.get_audio(data)
        return audio.detach().cpu().numpy()
