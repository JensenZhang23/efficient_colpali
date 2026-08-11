import torch
from math import ceil, sqrt
from typing import Any, Dict, List, Union, cast, Optional
from torch.nn.utils.rnn import pad_sequence

from PIL.Image import Image

from colpali_engine.models.idefics_2 import ColIdefics2Processor
from colpali_engine.models.paligemma import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import find_min_max_indices


class VisualRetrieverCollator:
    """
    Collator for training vision retrieval models.
    """

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
        pooling_strategy: Optional[str] = None,
        pool_size: Optional[int] = 1,
        factor: int = 28,
    ):
        self.processor = processor
        self.image_token_id = None
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.factor = factor
        self.kernel_size = int(sqrt(pool_size))

        if isinstance(self.processor, ColPaliProcessor) or isinstance(self.processor, ColIdefics2Processor):
            self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
                self.processor.tokenizer.additional_special_tokens.index("<image>")
            ]

        if isinstance(self.processor, ColPaliProcessor):
            if self.processor.tokenizer.padding_side != "right":
                print("Setting padding side to right")
                self.processor.tokenizer.padding_side = "right"

    def grid_index_to_patch_index(self, grid_index, new_height, new_width, grid_size=112, patch_size=28):
        """
        Convert grid index to patch index.
        """
        scale_factor = int(grid_size/patch_size)
        width_factor = new_width/grid_size
        i, j = grid_index // ceil(width_factor), grid_index % ceil(width_factor)

        patch_index_list = []
        first_token = width_factor*(scale_factor**2)*i + scale_factor*j
        for k in range(scale_factor):
            max_patch_index = min(
                (new_height//patch_size) * (new_width//patch_size),
                width_factor*scale_factor*(scale_factor*i+k+1)
            )
            for l in range(scale_factor):
                patch_index = int(first_token + k*(width_factor*scale_factor) + l)
                if patch_index < max_patch_index:
                    patch_index_list.append(patch_index)
        
        if len(patch_index_list)<self.kernel_size**2:
            patch_index_list += [-1]*(self.kernel_size**2-len(patch_index_list))
        return patch_index_list

    def __call__(
        self,
        examples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Collate function for the vision retriever associated to the collator's processor.
        """
        # Placeholders
        texts_query: Union[List[str], List[None], List[Union[str, None]]] = []  # some documents don't have a query
        images: List[Image] = []
        neg_images: List[Image] = []

        if self.processor is None or not isinstance(self.processor, BaseVisualRetrieverProcessor):
            raise ValueError("Processor should be provided for vision collator.")

        # Process each example
        for example in examples:
            texts_query.append(example["query"])
            if example["image"] is None:
                raise ValueError("Image is None - This collator does not support None images yet.")

            images.append(cast(Image, example["image"]))

            if "neg_image" in example and example["neg_image"] is not None:
                neg_images.append(cast(Image, example["neg_image"]))

        # Process the documents
        batch_doc = self.processor.process_images(
            images=images,
        )

        # Process the negative documents (if available)
        batch_neg_doc = None
        if len(neg_images) > 0:
            batch_neg_doc = self.processor.process_images(
                images=neg_images,
            )

        # Process the queries
        batch_query = None

        if all([t is None for t in texts_query]):
            # print("All queries are `None`. Returning `None` for all queries.")
            pass
        elif any([t is None for t in texts_query]):
            # If it's the first query that is not None but the rest are None, then it's hard negatives.
            raise ValueError("Some queries are None. This collator does not support None queries yet.")
        else:
            texts_query = cast(List[str], texts_query)
            batch_query = self.processor.process_queries(
                queries=texts_query,
                max_length=self.max_length,
            )

        # Prefix each key with "doc_" or "query_" to avoid key conflicts
        batch_all = {f"doc_{k}": v for k, v in batch_doc.items()}
        del batch_doc
        if batch_query is not None:
            batch_query = {f"query_{k}": v for k, v in batch_query.items()}
            batch_all.update(batch_query)
            del batch_query
        if batch_neg_doc is not None:
            batch_neg_doc = {f"neg_doc_{k}": v for k, v in batch_neg_doc.items()}
            batch_all.update(batch_neg_doc)

        patch_ranges_list = find_min_max_indices(batch_all["doc_input_ids"], self.processor.image_token_id)
        batch_all.update({"doc_patch_ranges": patch_ranges_list})
        
        if "doc_image_grid_thw" in batch_all:
            patch_indices_list = list()
            for grid_thw in batch_all["doc_image_grid_thw"]:
                token_num = torch.ceil(grid_thw[1]/(2*self.kernel_size)).long()*torch.ceil(grid_thw[2]/(2*self.kernel_size)).long()
                patch_indices = list()
                for i in range(token_num):
                    patch_index = self.grid_index_to_patch_index(
                        grid_index=i,
                        new_height=grid_thw[1]//2*self.factor,
                        new_width=grid_thw[2]//2*self.factor,
                        grid_size=self.factor*self.kernel_size,
                        patch_size=self.factor
                    )
                    patch_indices.append(patch_index)
                patch_indices = torch.tensor(patch_indices, dtype=torch.long)
                patch_indices_list.append(patch_indices)
            batch_all.update(
                {"doc_patch_indices_list": pad_sequence(patch_indices_list, batch_first=True, padding_value=-1)}
            )

        return batch_all
