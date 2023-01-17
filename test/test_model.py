from legup.train.models.anymal.teacher import Teacher

import torch


def test_confirm_teacher_dimensions():
    """
    Confirm that the teacher dimensions are correct.
    """
    teacher = Teacher(None)
    actor, critic = teacher(torch.rand([1, 391]))
    assert actor.shape == torch.Size([1, 16]), "actor should be shape (1, 16), but is {actor_shape}".format(
        actor_shape=actor.shape)

    assert critic.shape == torch.Size([1, 1]), "critic should be shape (1, 1), but is {critic_shape}".format(
        critic_shape=critic.shape)

    actor, critic = teacher(torch.rand([2, 391]))
    assert actor.shape == torch.Size([2, 16]), "actor should be shape (2, 16), but is {actor_shape}".format(
        actor_shape=actor.shape)

    assert critic.shape == torch.Size([2, 1]), "critic should be shape (2, 1), but is {critic_shape}".format(
        critic_shape=critic.shape)

    actor, critic = teacher(torch.rand([1000, 391]))
    assert actor.shape == torch.Size([1000, 16]), "actor should be shape (1000, 16), but is {actor_shape}".format(
        actor_shape=actor.shape)

    assert critic.shape == torch.Size([1000, 1]), "critic should be shape (1000, 1), but is {critic_shape}".format(
        critic_shape=critic.shape)
